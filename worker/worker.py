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
import pickle
import math

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, source, name, models_loc, start, n_steps, test=False):

        while True:
            # putting this while loop here because sometime the optimizer
            # is writing to the files and causes an exception
            self.market_encoder = AttentionMarketEncoder()
            self.encoder_to_others = EncoderToOthers()
            self.proposer = Proposer()
            self.actor_critic = ActorCritic()
            self.encoder_to_others = EncoderToOthers()
            try:
                # this is the lstm's version
                # self.market_encoder = MarketEncoder()
                # this is the attention version
                self.market_encoder.load_state_dict(torch.load(models_loc + 'market_encoder.pt'))
                self.encoder_to_others.load_state_dict(torch.load(models_loc + 'encoder_to_others.pt'))
                self.proposer.load_state_dict(torch.load(models_loc + 'proposer.pt'))
                self.actor_critic.load_state_dict(torch.load(models_loc + 'actor_critic.pt'))
                self.market_encoder = self.market_encoder.cpu()
                self.proposer = self.proposer.cpu()
                self.actor_critic = self.actor_critic.cpu()
                self.encoder_to_others = self.encoder_to_others.cpu()
                break
            except Exception:
                pass
        self.models_loc = models_loc

        self.server = redis.Redis("localhost")
        self.name = name
        self.proposed_sigma = float(self.server.get("proposed_sigma_" + name).decode("utf-8"))
        self.policy_sigma = float(self.server.get("policy_sigma_" + name).decode("utf-8"))
        self.spread_func_param = float(self.server.get("spread_func_param_" + name).decode("utf-8"))

        self.window = networks.WINDOW
        self.n_steps = n_steps

        self.environment = Env(source, start, n_steps, self.spread_func_param, self.window, get_time=True)

        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.reward_tau = float(self.server.get("reward_tau").decode("utf-8"))

        self.test = test

    def run(self):
        all_rewards = 0

        time_states = []
        percents_in = []
        spreads = []
        proposed_actions = []
        mus = []
        actions = []
        rewards = []

        state = self.environment.get_state()
        time_states_, percent_in_, spread_, reward_ = state
        time_states = time_states_
        t0 = time.time()
        for i_step in range(self.n_steps):

            # try getting the latest versions of the models
            try:
                self.market_encoder.load_state_dict(torch.load(self.models_loc + 'market_encoder.pt'))
                self.proposer.load_state_dict(torch.load(self.models_loc + 'proposer.pt'))
                self.actor_critic.load_state_dict(torch.load(self.models_loc + 'actor_critic.pt'))
                self.market_encoder = self.market_encoder.cpu()
                self.proposer = self.proposer.cpu()
                self.actor_critic = self.actor_critic.cpu()
            except Exception:
                pass

            input_time_states = torch.cat(time_states[-self.window:]).cpu()
            mean = input_time_states[:, 0, :4].mean()
            std = input_time_states[:, 0, :4].std()
            input_time_states[:, 0, :4] = (input_time_states[:, 0, :4] - mean) / std
            spread_normalized = spread_ / std
            # this is the lstm's version
            # market_encoding = self.market_encoder.forward(input_time_states, torch.Tensor([percent_in_]).cpu(), torch.Tensor([spread_]).cpu(), 'cpu')
            # this is the attention version
            market_encoding = self.market_encoder.forward(input_time_states)
            percents_in.append(percent_in_)
            spreads.append(spread_)

            market_encoding = self.encoder_to_others.forward(market_encoding, torch.Tensor([spread_normalized]).cpu(), torch.Tensor([percent_in_]).cpu())
            queried_actions = self.proposer.forward(market_encoding, exploration_parameter=torch.randn(1, 2).cpu() * self.proposed_sigma).cpu()
            proposed_actions.append(queried_actions)

            policy, value = self.actor_critic.forward(market_encoding, queried_actions, sigma=self.policy_sigma)

            action = int(torch.multinomial(policy, 1))
            mu = policy[0, action]
            actions.append(action)
            mus.append(mu)

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
                                        r=all_rewards))
                # print("step:", i_step)
                # print("queried_actions:", queried_actions)
                # print("policy:", policy)
                # print("value:", value * reward_emsd + reward_ema)
                # print("sum rewards:", all_rewards)
                # print("normalized sum rewards:", (all_rewards - reward_ema) / (reward_emsd + 1e-9))

            # adding steps_since_trade to help it learn
            if action in [0, 1]:
                placed_order = [action, float(queried_actions[0, action])]
                steps_since_trade = 0
            else:
                placed_order = [action]

            self.environment.step(placed_order)

            state = self.environment.get_state()
            if not state:
                break

            time_states_, percent_in_, spread_, reward_ = state
            if not self.test:
                reward_ema = self.server.get("reward_ema").decode("utf-8")
                if reward_ema == 'None':
                    self.server.set("reward_ema", reward_)
                    self.server.set("reward_emsd", 0)
                else:
                    reward_ema = float(reward_ema)
                    reward_emsd = float(self.server.get("reward_emsd").decode("utf-8"))
                    delta = reward_ - reward_ema
                    self.server.set("reward_ema", reward_ema + self.reward_tau * delta)
                    self.server.set("reward_emsd", math.sqrt((1 - self.reward_tau) * (reward_emsd**2 + self.reward_tau * (delta**2))))

            time_states.append(time_states_[-1])
            rewards.append(reward_)
            all_rewards += reward_

        if not self.test:
            experience = Experience(time_states, percents_in, spreads, mus, proposed_actions, actions, rewards)
            if not self.test:
                self.server.rpush("experience", pickle.dumps(experience, protocol=pickle.HIGHEST_PROTOCOL))

        print("name: {name}, steps: {steps}, time: {time}, sum all rewards = {reward}".format(name=self.name, steps=i_step, time=time.time()-t0, reward=np.sum(all_rewards)))
        return all_rewards

Experience = namedtuple('Experience', ('time_states',
                                       'percents_in',
                                       'spreads',
                                       'mus',
                                       'proposed_actions',
                                       'place_actions',
                                       'rewards'))

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    server = redis.Redis("localhost")
    server.set("proposed_sigma_test", 0)
    server.set("policy_sigma_test", 1)
    server.set("spread_func_param_test", 0)
    source = "../normalized_data/DAT_MT_EURUSD_M1_2010-1.3261691621962404.csv"
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_loc = dir_path + '/../models/'
    start = np.random.randint(0, 200000)
    # start = 0
    n_steps = 1_000_000
    n_steps = 7200
    # n_steps = int(server.get("trajectory_steps").decode("utf-8"))
    test = True
    i = 1
    sum_rewards = 0
    while True:
        worker = Worker(source, "test", models_loc, start, n_steps, test)
        sum_rewards += worker.run()
        print("AVERAGE RETURN:", sum_rewards / i)
        print("NUMBER OF SAMPLES:", i)
        time.sleep(2)
        i += 1
