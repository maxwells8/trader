import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
import networks
from networks import *
from environment import *
import redis
import msgpack
import math
from zeus.zeus import Zeus

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, name, instrument, granularity, models_loc, start, zeta, test=False):

        while True:
            # putting this while loop here because sometime the optimizer
            # is writing to the files and causes an exception
            self.market_encoder = CNNEncoder().cpu()
            self.encoder_to_others = EncoderToOthers().cpu()
            self.proposer = Proposer().cpu()
            self.actor_critic = ActorCritic().cpu()
            self.encoder_to_others = EncoderToOthers().cpu()
            try:
                self.market_encoder.load_state_dict(torch.load(models_loc + 'market_encoder.pt'))
                self.encoder_to_others.load_state_dict(torch.load(models_loc + 'encoder_to_others.pt'))
                self.proposer.load_state_dict(torch.load(models_loc + 'proposer.pt'))
                self.actor_critic.load_state_dict(torch.load(models_loc + 'actor_critic.pt'))
                self.market_encoder = self.market_encoder.cpu()
                self.encoder_to_others = self.encoder_to_others.cpu()
                self.proposer = self.proposer.cpu()
                self.actor_critic = self.actor_critic.cpu()
                break
            except Exception as e:
                print(e)

        self.name = name
        self.models_loc = models_loc

        self.server = redis.Redis("localhost")
        self.zeus = Zeus(instrument, granularity)

        self.time_states = []
        self.percents_in = []
        self.spreads = []
        self.proposed_actions = []
        self.mus = []
        self.actions = []
        self.rewards = []

        self.total_actual_reward = 0

        # zeta is a way to disincentivize swtiching between buying and selling.
        # it's given a negative reward of zeta each time it switches.
        self.zeta = zeta

        self.reward_tau = float(self.server.get("reward_tau").decode("utf-8"))
        self.proposed_sigma = float(self.server.get("proposed_sigma_" + name).decode("utf-8"))
        self.policy_sigma = float(self.server.get("policy_sigma_" + name).decode("utf-8"))
        self.spread_reimbursement_ratio = float(self.server.get("spread_reimbursement_ratio").decode("utf-8"))

        self.window = networks.WINDOW

        self.start = start
        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.test = test
        self.steps_before_trajectory = np.random.randint(0, 360)
        self.n_steps_left = self.window + self.trajectory_steps + self.steps_before_trajectory

        self.prev_value = self.zeus.unrealized_balance()

    def add_bar(self, bar):
        time_state = [[[bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]]]

        self.time_states.append(time_state)

        if len(self.time_states) >= self.window:
            percent_in = self.zeus.position_size() / (abs(self.zeus.position_size()) + self.zeus.units_available() + 1e-9)

            input_time_states = torch.Tensor(self.time_states[-self.window:]).view(self.window, 1, networks.D_BAR).cpu()
            mean = input_time_states[:, 0, :4].mean()
            std = input_time_states[:, 0, :4].std()
            input_time_states[:, 0, :4] = (input_time_states[:, 0, :4] - mean) / std
            spread_normalized = bar.spread / std

            market_encoding = self.market_encoder.forward(input_time_states)
            market_encoding = self.encoder_to_others.forward(market_encoding, (std + 1e-9).log(), torch.Tensor([spread_normalized]), torch.Tensor([percent_in]))
            queried_actions = self.proposer.forward(market_encoding, exploration_parameter=torch.randn(1, 2) * self.proposed_sigma)
            policy, value = self.actor_critic.forward(market_encoding, (queried_actions + 1e-9).log(), sigma=self.policy_sigma)

            before = self.zeus.unrealized_balance()
            reward = 0
            action = torch.multinomial(policy, 1).item()
            if action == 0:
                if self.zeus.position_size() < 0:
                    reward -= self.zeta
                    amount = int(abs(self.zeus.position_size()))
                    self.zeus.close_units(amount)
                desired_percent_in = queried_actions[0, 0].item()
                current_percent_in = abs(self.zeus.position_size()) / (abs(self.zeus.position_size()) + self.zeus.units_available() + 1e-9)
                diff_percent = desired_percent_in - current_percent_in
                if diff_percent < 0:
                    # print(0, 0, desired_percent_in, diff_percent, self.zeus.position_size(), self.zeus.units_available())
                    amount = int(abs(diff_percent * (abs(self.zeus.position_size()) + self.zeus.units_available())))
                    # print(amount)
                    self.zeus.close_units(amount)
                else:
                    # print(0, 1, desired_percent_in, diff_percent, self.zeus.position_size(), self.zeus.units_available())
                    amount = int(diff_percent * max(abs(self.zeus.position_size()) + self.zeus.units_available(), 0))
                    # print(amount)
                    self.zeus.place_trade(amount, "Long")
            elif action == 1:
                if self.zeus.position_size() > 0:
                    reward -= self.zeta
                    amount = int(abs(self.zeus.position_size()))
                    self.zeus.close_units(amount)
                desired_percent_in = queried_actions[0, 1].item()
                current_percent_in = abs(self.zeus.position_size()) / (abs(self.zeus.position_size()) + self.zeus.units_available() + 1e-9)
                diff_percent = desired_percent_in - current_percent_in
                if diff_percent < 0:
                    # print(1, 0, desired_percent_in, diff_percent, self.zeus.position_size(), self.zeus.units_available())
                    amount = int(abs(diff_percent * (abs(self.zeus.position_size()) + self.zeus.units_available())))
                    # print(amount)
                    self.zeus.close_units(amount)
                else:
                    # print(1, 1, desired_percent_in, diff_percent, self.zeus.position_size(), self.zeus.units_available())
                    amount = int(diff_percent * max(abs(self.zeus.position_size()) + self.zeus.units_available(), 0))
                    # print(amount)
                    self.zeus.place_trade(amount, "Short")
            # else:
                # # to disincentivize holding
                # reward -= 0.1
            reward += (before - self.zeus.unrealized_balance()) * self.spread_reimbursement_ratio
            self.total_actual_reward += self.zeus.unrealized_balance() - self.prev_value
            reward += self.zeus.unrealized_balance() - self.prev_value
            self.prev_value = self.zeus.unrealized_balance()
            # print(action, reward)
            mu = policy[0, action].item()

            if self.test:
                reward_ema = float(self.server.get("reward_ema").decode("utf-8"))
                reward_emsd = float(self.server.get("reward_emsd").decode("utf-8"))
                print("step: {s} \
                \nqueried_actions: {q} \
                \npolicy: {p} \
                \nvalue: {v} \
                \nrewards: {r}\n".format(s=i_step,
                                        q=queried_actions,
                                        p=policy,
                                        v=value,
                                        r=np.sum(self.rewards)))
                # print("step:", i_step)
                # print("queried_actions:", queried_actions)
                # print("policy:", policy)
                # print("value:", value * reward_emsd + reward_ema)
                # print("sum rewards:", np.sum(self.rewards))
                # print("normalized sum rewards:", (np.sum(self.rewards) - reward_ema) / (reward_emsd + 1e-9))

            # adding steps_since_trade to help it learn
            if action in [0, 1]:
                placed_order = [action, queried_actions[0, action].item()]
                steps_since_trade = 0
            else:
                placed_order = [action]

            if not self.test:
                reward_ema = self.server.get("reward_ema").decode("utf-8")
                if reward_ema == 'None':
                    self.server.set("reward_ema", reward)
                    self.server.set("reward_emsd", 0)
                else:
                    reward_ema = float(reward_ema)
                    reward_emsd = float(self.server.get("reward_emsd").decode("utf-8"))
                    delta = reward - reward_ema
                    self.server.set("reward_ema", reward_ema + self.reward_tau * delta)
                    self.server.set("reward_emsd", math.sqrt((1 - self.reward_tau) * (reward_emsd**2 + self.reward_tau * (delta**2))))

            self.percents_in.append(percent_in)
            self.spreads.append(bar.spread)
            self.rewards.append(reward)
            self.actions.append(action)
            self.mus.append(mu)
            self.proposed_actions.append(queried_actions.tolist())

            if len(self.time_states) == self.window + self.trajectory_steps + 1:
                del self.time_states[0]
            if len(self.percents_in) == self.trajectory_steps + 1:
                del self.percents_in[0]
                del self.spreads[0]
                del self.rewards[0]
                del self.actions[0]
                del self.mus[0]
                del self.proposed_actions[0]

            if not self.test and len(self.time_states) == self.window + self.trajectory_steps:
                experience = Experience(
                self.time_states,
                self.percents_in,
                self.spreads,
                self.mus,
                self.proposed_actions,
                self.actions,
                self.rewards
                )
                experience = msgpack.packb(experience, use_bin_type=True)
                n_experiences = self.server.llen("experience")
                if n_experiences > 0:
                    loc = np.random.randint(0, n_experiences)
                    ref = self.server.lindex("experience", loc)
                    self.server.linsert("experience", "before", ref, experience)
                else:
                    self.server.lpush("experience", experience)

        self.n_steps_left -= 1
        torch.cuda.empty_cache()

    def run(self):
        t0 = time.time()
        while self.n_steps_left > 0:
            n_seconds = self.n_steps_left * 60
            self.zeus.stream_range(self.start, self.start + n_seconds, self.add_bar)
            self.start += n_seconds
        print("time: {time}, total rewards: {reward}, steps: {steps}".format(time=time.time()-t0, reward=self.total_actual_reward, steps=self.steps_before_trajectory))
        return self.total_actual_reward

Experience = namedtuple('Experience', ('time_states',
                                       'percents_in',
                                       'spreads',
                                       'mus',
                                       'proposed_actions',
                                       'place_actions',
                                       'rewards'))
