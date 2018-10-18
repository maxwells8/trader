import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
from networks import *
from environment import *
import redis
import pickle
import math

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

"""
the worker acts in the environment with a fixed policy for a certain amount of
time, and it also prepares the experience for the optimizer.
"""
class Worker(object):

    def __init__(self, source, name, models_loc, window, start, n_steps, test=False):

        while True:
            # putting this while loop here because sometime the optimizer
            # is writing to the files and causes an exception
            try:
                # this is the lstm's version
                # self.market_encoder = MarketEncoder()
                # this is the attention version
                self.market_encoder = AttentionMarketEncoder()
                self.proposer = Proposer()
                self.actor_critic = ActorCritic()
                self.market_encoder.load_state_dict(torch.load(models_loc + '/market_encoder.pt'))
                self.proposer.load_state_dict(torch.load(models_loc + '/proposer.pt'))
                self.actor_critic.load_state_dict(torch.load(models_loc + '/actor_critic.pt'))
                self.market_encoder = self.market_encoder.cpu()
                self.proposer = self.proposer.cpu()
                self.actor_critic = self.actor_critic.cpu()
                break
            except Exception:
                pass
        self.models_loc = models_loc

        self.server = redis.Redis("localhost")
        self.name = name
        self.proposed_sigma = float(self.server.get("proposed_sigma_" + name).decode("utf-8"))
        self.policy_sigma = float(self.server.get("policy_sigma_" + name).decode("utf-8"))
        self.spread_func_param = float(self.server.get("spread_func_param_" + name).decode("utf-8"))

        self.environment = Env(source, start, n_steps, self.spread_func_param, window, get_time=True)

        self.window = window
        self.n_steps = n_steps

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
        initial_time_states, initial_percent_in, initial_spread, _ = state
        initial_time_states = torch.cat(initial_time_states).cpu()
        mean = initial_time_states[:, 0, :4].mean()
        std = initial_time_states[:, 0, :4].std()
        initial_time_states[:, 0, :4] = (initial_time_states[:, 0, :4] - mean) / std
        initial_spread = initial_spread / std
        t0 = time.time()
        for i_step in range(self.n_steps):

            if i_step >= self.window:

                # try getting the latest versions of the models
                try:
                    self.market_encoder.load_state_dict(torch.load(self.models_loc + '/market_encoder.pt'))
                    self.proposer.load_state_dict(torch.load(self.models_loc + '/proposer.pt'))
                    self.actor_critic.load_state_dict(torch.load(self.models_loc + '/actor_critic.pt'))
                    self.market_encoder = self.market_encoder.cpu()
                    self.proposer = self.proposer.cpu()
                    self.actor_critic = self.actor_critic.cpu()
                except Exception:
                    pass

                time_states.append(initial_time_states)
                percents_in.append(initial_percent_in)
                spreads.append(initial_spread)

                # this is the lstm's version
                # market_encoding = self.market_encoder.forward(initial_time_states, torch.Tensor([initial_percent_in]).cpu(), torch.Tensor([initial_spread]).cpu(), 'cpu')
                # this is the attention version
                market_encoding = self.market_encoder.forward(initial_time_states, torch.Tensor([initial_percent_in]).cpu(), torch.Tensor([initial_spread]).cpu())

                queried_actions = self.proposer.forward(market_encoding, torch.randn(1, 2).cpu() * self.proposed_sigma).cpu()
                proposed_actions.append(queried_actions)

                policy, value = self.actor_critic.forward(market_encoding, queried_actions, self.policy_sigma)

                action = int(torch.multinomial(policy, 1))
                mu = policy[0, action]
                actions.append(action)
                mus.append(mu)

                if self.test:
                    time.sleep(0.025)
                    reward_ema = float(self.server.get("reward_ema").decode("utf-8"))
                    reward_emsd = float(self.server.get("reward_emsd").decode("utf-8"))
                    print("step:", i_step)
                    print("queried_actions:", queried_actions)
                    print("policy:", policy)
                    print("value:", value * reward_emsd + reward_ema)
                    print("sum rewards:", all_rewards)
                    print("normalized sum rewards:", (all_rewards - reward_ema) / reward_emsd)

                # adding steps_since_trade to help it learn
                if action == 0:
                    placed_order = [0, float(queried_actions[0, 0])]
                    steps_since_trade = 0
                elif action == 1:
                    placed_order = [1, float(queried_actions[0, 1])]
                    steps_since_trade = 0
                else:
                    placed_order = [action]

                self.environment.step(placed_order)

            else:
                time_states.append(None)
                percents_in.append(None)
                proposed_actions.append(None)
                actions.append(None)
                mus.append(None)
                self.environment.step([3])

            state = self.environment.get_state()
            if not state:
                break

            final_time_states, final_percent_in, final_spread, reward = state
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

            final_time_states = torch.cat(final_time_states).cpu()
            mean = final_time_states[:, 0, :4].mean()
            std = final_time_states[:, 0, :4].std()
            final_time_states[:, 0, :4] = (final_time_states[:, 0, :4] - mean) / std
            final_spread = final_spread / std

            rewards.append(reward)
            all_rewards += reward

            if i_step >= self.trajectory_steps + self.window - 1:
                experience = Experience(time_states + [final_time_states], percents_in + [final_percent_in], spreads + [final_spread], mus, proposed_actions, actions, rewards)
                if not self.test:
                    self.server.rpush("experience", pickle.dumps(experience, protocol=pickle.HIGHEST_PROTOCOL))

            initial_time_states = final_time_states
            initial_percent_in = final_percent_in
            initial_spread = final_spread

            if len(time_states) == self.trajectory_steps:
                del time_states[0]
                del percents_in[0]
                del mus[0]
                del proposed_actions[0]
                del actions[0]
                del rewards[0]

        print("name: {name}, steps: {steps}, time: {time}, sum all rewards = {reward}".format(name=self.name, steps=i_step, time=time.time()-t0, reward=np.sum(all_rewards)))


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
    source = "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2010-1.3261691621962404.csv"
    models_loc = '../models'
    window = 128
    start = np.random.randint(0, 200000)
    start = 0
    n_steps = 1_000_000
    n_steps = window + 128
    test = True
    while True:
        worker = Worker(source, "test", models_loc, window, start, n_steps, test)
        worker.run()
