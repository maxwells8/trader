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

    def __init__(self, name, instrument, granularity, models_loc, start, test=False):

        while True:
            self.market_encoder = LSTMCNNEncoder().cpu()
            self.encoder_to_others = EncoderToOthers().cpu()
            self.proposer = ProbabilisticProposer().cpu()
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
            except Exception:
                print("Failed to load models")
                time.sleep(0.1)

        self.name = name
        self.models_loc = models_loc

        self.server = redis.Redis("localhost")
        self.instrument = instrument
        self.zeus = Zeus(instrument, granularity)

        self.time_states = []
        self.percents_in = []
        self.spreads = []
        self.proposed_actions = []
        self.proposed_mus = []
        self.mus = []
        self.actions = []
        self.rewards = []

        self.total_actual_reward = 0
        self.reward_tau = float(self.server.get("reward_tau").decode("utf-8"))

        self.window = networks.WINDOW

        self.start = start
        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.p_new_proposal = float(self.server.get("p_new_proposal").decode("utf-8"))

        self.tradeable_percentage = 0.1

        self.test = test
        if self.test:
            self.steps_before_trajectory = 1440 - self.trajectory_steps
            self.actor_temp = 1
        else:
            self.steps_before_trajectory = 500
            base_temp = float(self.server.get("actor_temp").decode("utf-8"))
            self.actor_temp = np.random.exponential(base_temp / np.log(2))
        self.n_steps_left = self.window + self.trajectory_steps + self.steps_before_trajectory
        self.i_step = 0
        self.steps_since_push = 0
        self.steps_between_experiences = 2

        self.prev_value = self.zeus.unrealized_balance()
        self.prev_queried = None

    def add_bar(self, bar):
        time_state = [[[bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]]]

        if len(self.time_states) == 0 or time_state != self.time_states[-1]:
            self.time_states.append(time_state)
        else:
            return

        if len(self.time_states) >= self.window:
            percent_in = (self.zeus.position_size() / (abs(self.zeus.position_size()) + self.zeus.units_available() + 1e-9)) / self.tradeable_percentage

            self.total_actual_reward += self.zeus.unrealized_balance() - self.prev_value
            reward = (self.zeus.unrealized_balance() - self.prev_value) / (2000 * self.tradeable_percentage)
            self.prev_value = self.zeus.unrealized_balance()

            input_time_states = torch.Tensor(self.time_states[-self.window:]).view(self.window, 1, networks.D_BAR).cpu()
            mean = input_time_states[:, 0, :4].mean()
            std = input_time_states[:, 0, :4].std()
            input_time_states[:, 0, :4] = (input_time_states[:, 0, :4] - mean) / std
            assert torch.isnan(input_time_states).sum() == 0
            spread_normalized = bar.spread / std

            market_encoding = self.market_encoder.forward(input_time_states)
            market_encoding = self.encoder_to_others.forward(market_encoding, torch.Tensor([spread_normalized]), torch.Tensor([percent_in]))

            if np.random.rand() < self.p_new_proposal or len(self.time_states) == self.window:
                queried_actions, p_actions = self.proposer.forward(market_encoding)
                self.prev_queried = queried_actions.detach()
            else:
                queried_actions = self.prev_queried
                _, _, p_w, p_mu, p_sigma = self.proposer.forward(market_encoding, True)
                p_actions = self.proposer.p(queried_actions, p_w, p_mu, p_sigma)
            p_actions = p_actions.prod(1).view(-1, 1)

            policy, value = self.actor_critic.forward(market_encoding, queried_actions, self.actor_temp)

            if self.test:
                action = torch.argmax(policy).item()
                # action = torch.multinomial(policy, 1).item()
                # action = np.random.randint(0, 2)
            else:
                action = torch.multinomial(policy, 1).item()

            mu = policy[0, action].item()
            action_mu = p_actions.item() + 1e-9

            if action == 0:
                if self.zeus.position_size() < 0:
                    amount = int(abs(self.zeus.position_size()))
                    self.zeus.close_units(amount)
                desired_percent_in = queried_actions[0, 0].item() * self.tradeable_percentage
                current_percent_in = abs(self.zeus.position_size()) / (abs(self.zeus.position_size()) + self.zeus.units_available() + 1e-9)
                diff_percent = desired_percent_in - current_percent_in
                if diff_percent < 0:
                    amount = int(abs(diff_percent * (abs(self.zeus.position_size()) + self.zeus.units_available())))
                    self.zeus.close_units(amount)
                else:
                    amount = int(diff_percent * (abs(self.zeus.position_size()) + self.zeus.units_available()))
                    self.zeus.place_trade(amount, "Long")

            elif action == 1:
                if self.zeus.position_size() > 0:
                    amount = int(abs(self.zeus.position_size()))
                    self.zeus.close_units(amount)
                desired_percent_in = queried_actions[0, 1].item() * self.tradeable_percentage
                current_percent_in = abs(self.zeus.position_size()) / (abs(self.zeus.position_size()) + self.zeus.units_available() + 1e-9)
                diff_percent = desired_percent_in - current_percent_in
                if diff_percent < 0:
                    amount = int(abs(diff_percent * (abs(self.zeus.position_size()) + self.zeus.units_available())))
                    self.zeus.close_units(amount)
                else:
                    amount = int(diff_percent * (abs(self.zeus.position_size()) + self.zeus.units_available()))
                    self.zeus.place_trade(amount, "Short")

            # else:
            #     amount = int(abs(self.zeus.position_size()))
            #     self.zeus.close_units(amount)
            #     pass


            if self.test:
                # time.sleep(0.1)
                reward_ema = self.server.get("test_reward_ema")
                reward_emsd = self.server.get("test_reward_emsd")
                if reward_ema != None:
                    reward_ema = float(reward_ema.decode("utf-8"))
                    reward_emsd = float(reward_emsd.decode("utf-8"))
                else:
                    reward_ema = 0
                    reward_emsd = 0

                print("step: {s} \
                \npercent in: {p_in} \
                \naction: {a} \
                \nunrealized_balance: {u_b} \
                \nproposed: {prop} \
                \nproposed probability: {prop_p} \
                \npolicy: {p} \
                \nvalue: {v} \
                \nrewards: {r} \
                \nreward_ema: {ema} \
                \nreward_emsd: {emsd}\n".format(s=self.i_step,
                                        p_in=round(percent_in, 8),
                                        a=action,
                                        u_b=round(self.zeus.unrealized_balance(), 2),
                                        prop=queried_actions,
                                        prop_p=action_mu,
                                        p=policy,
                                        v=round(value.item(), 2),
                                        r=round(self.total_actual_reward, 2),
                                        ema=round(reward_ema, 2),
                                        emsd=round(reward_emsd, 2)))

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
            self.proposed_mus.append(action_mu)
            self.proposed_actions.append(queried_actions.tolist())

            if len(self.time_states) == self.window + self.trajectory_steps + 1:
                del self.time_states[0]
            if len(self.percents_in) == self.trajectory_steps + 1:
                del self.percents_in[0]
                del self.spreads[0]
                del self.rewards[0]
                del self.actions[0]
                del self.mus[0]
                del self.proposed_mus[0]
                del self.proposed_actions[0]

            if (self.steps_since_push >= self.steps_between_experiences) and not self.test and len(self.time_states) == self.window + self.trajectory_steps:
                experience = Experience(
                self.time_states,
                self.percents_in,
                self.spreads,
                self.mus,
                self.proposed_mus,
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
                self.steps_since_push = 0
            else:
                self.steps_since_push += 1

            self.i_step += 1

        self.n_steps_left -= 1
        torch.cuda.empty_cache()

    def run(self):
        self.market_encoder.eval()
        self.encoder_to_others.eval()
        self.proposer.eval()
        self.actor_critic.eval()

        t0 = time.time()
        while self.n_steps_left > 0:
            n_seconds = min(self.n_steps_left, 500) * 60
            self.zeus.stream_range(self.start, self.start + n_seconds, self.add_bar)
            self.start += n_seconds
        print("time: {time}, total rewards: {reward}, actor temp: {actor_temp}, steps: {steps}".format(time=round(time.time()-t0, 2), reward=round(self.total_actual_reward, 2), actor_temp=round(self.actor_temp, 5), steps=self.steps_before_trajectory))
        if self.test:
            reward_tau = float(self.server.get("test_reward_tau").decode("utf-8"))
            reward_ema = self.server.get("test_reward_ema")
            if reward_ema == None:
                self.server.set("test_reward_ema", self.total_actual_reward)
                self.server.set("test_reward_emsd", 0)
            else:
                reward_ema = float(reward_ema.decode("utf-8"))
                reward_emsd = float(self.server.get("test_reward_emsd").decode("utf-8"))
                delta = self.total_actual_reward - reward_ema
                self.server.set("test_reward_ema", reward_ema + reward_tau * delta)
                self.server.set("test_reward_emsd", math.sqrt((1 - reward_tau) * (reward_emsd**2 + reward_tau * (delta**2))))

        return self.total_actual_reward

Experience = namedtuple('Experience', ('time_states',
                                       'percents_in',
                                       'spreads',
                                       'mus',
                                       'proposed_mus',
                                       'proposed_actions',
                                       'place_actions',
                                       'rewards'))
