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

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

"""
the worker acts in the environment with a fixed policy for a certain amount of
time, and it also prepares the experience for the optimizer.
"""
class Worker(object):

    def __init__(self, source, name, models_loc, window, n_steps, test=False):

        self.environment = Env(source, time_window=window)
        self.market_encoder = torch.load(models_loc + '/market_encoder.pt').cpu()
        self.proposer = torch.load(models_loc + '/proposer.pt').cpu()
        self.actor_critic = torch.load(models_loc + '/actor_critic.pt').cpu()
        self.models_loc = models_loc

        self.server = redis.Redis("localhost")
        self.name = name
        self.sigma = float(self.server.get("sigma_" + name).decode("utf-8"))

        self.window = window
        self.n_steps = n_steps

        self.test = test

    def run(self):
        reward_ema = 0
        ema_parameter = 0.01
        rewards = []
        state = self.environment.get_state()
        initial_time_states, initial_percent_in, _ = state
        initial_time_states = torch.cat(initial_time_states).cpu()
        mean = initial_time_states[:, 0, :4].mean()
        std = initial_time_states[:, 0, :4].std()
        initial_time_states[:, 0, :4] = (initial_time_states[:, 0, :4] - mean) / std
        initial_time_states[:, 0, 5] = initial_time_states[:, 0, 5] / std
        for i_step in range(self.n_steps):

            market_encoding = self.market_encoder.forward(initial_time_states, torch.Tensor([initial_percent_in]).cpu(), 'cpu')

            queried_actions = self.proposer.forward(market_encoding, torch.randn(1, 2).cpu() * self.sigma)

            policy, value = self.actor_critic.forward(market_encoding, queried_actions)

            if self.test:
                print("queried_actions:", queried_actions)
                print("(policy, value):", policy, value)
                print("reward_ema:", reward_ema)
                print("past 100 rewards:", np.sum(rewards))

            action = int(torch.multinomial(policy, 1))
            mu = policy[0, action]

            if action == 0:
                placed_order = [0, float(queried_actions[0, 0])]
            elif action == 1:
                placed_order = [1, float(queried_actions[0, 1])]
            else:
                placed_order = [2]

            self.environment.step(placed_order)

            state = self.environment.get_state()
            if not state:
                break

            final_time_states, final_percent_in, reward = state
            final_time_states = torch.cat(final_time_states).cpu()
            mean = final_time_states[:, 0, :4].mean()
            std = final_time_states[:, 0, :4].std()
            final_time_states[:, 0, :4] = (final_time_states[:, 0, :4] - mean) / std
            final_time_states[:, 0, 5] = final_time_states[:, 0, 5] / std

            reward_ema = ema_parameter * reward + (1 - ema_parameter) * reward_ema
            rewards.append(reward)
            if len(rewards) > 100:
                rewards.pop(0)

            if len(initial_time_states) == self.window:
                experience = Experience(initial_time_states, initial_percent_in, mu, queried_actions, action, reward, final_time_states, final_percent_in)
                if not self.test:
                    self.server.rpush("experience", pickle.dumps(experience, protocol=pickle.HIGHEST_PROTOCOL))

            initial_time_states = final_time_states
            initial_percent_in = final_percent_in

            if i_step % 100 == 0:
                print("{name} after {steps} steps: ema = {ema}, sum past 100 rewards = {reward}".format(name=self.name, steps=i_step, ema=reward_ema, reward=np.sum(rewards)))

            if i_step % 100 == 0:
                # doing this while loop in case the models are being written to while trying to read them
                while True:
                    try:
                        self.market_encoder = torch.load(self.models_loc + '/market_encoder.pt').cpu()
                        self.proposer = torch.load(self.models_loc + '/proposer.pt').cpu()
                        self.actor_critic = torch.load(self.models_loc + '/actor_critic.pt').cpu()
                        break
                    except Exception:
                        pass
                self.environment.reset()
                state = self.environment.get_state()
                if not state:
                    break
                initial_time_states, initial_percent_in, _ = state
                initial_time_states = torch.cat(initial_time_states).cpu()
                mean = initial_time_states[:, 0, :4].mean()
                std = initial_time_states[:, 0, :4].std()
                initial_time_states[:, 0, :4] = (initial_time_states[:, 0, :4] - mean) / std
                initial_time_states[:, 0, 5] = initial_time_states[:, 0, 5] / std


Experience = namedtuple('Experience', ('initial_time_states',
                                       'initial_percent_in',
                                       'mu',
                                       'proposed_actions',
                                       'place_action',
                                       'reward',
                                       'final_time_states',
                                       'final_percent_in'))

if __name__ == "__main__":
    server = redis.Redis("localhost")
    server.set("sigma_3", 0)
    source = "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2017-1.1294884577273274.csv"
    models_loc = '../models'
    window = 256
    n_steps = 1000000

    worker = Worker(source, "3", models_loc, window, n_steps, True)
    worker.run()
