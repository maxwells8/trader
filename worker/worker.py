import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from collections import namedtuple
from networks import *
from environment import *
import redis
import pickle

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)

"""
TODO
- change all the devices to cpu
- update to the new network architecture
"""
class Worker(object):

    def __init__(self, source, name, models_loc, window):

        self.environment = Env(source, time_window=window)
        self.market_encoder = torch.load(models_loc + '/market_encoder.pt').cpu()
        self.proposer = torch.load(models_loc + '/proposer.pt').cpu()
        self.actor_critic = torch.load(models_loc + '/actor_critic.pt').cpu()

        self.server = redis.Redis("localhost")
        self.name = name
        self.sigma = float(self.server.get("sigma_" + name).decode("utf-8"))

        self.experience = []

    def run(self):
        value = self.environment.value

        replay_initial_time_states = None
        replay_initial_percent_in = None
        replay_final_time_states = None
        replay_final_percent_in = None
        replay_proposed = None
        replay_place_action = None
        replay_mu = None
        replay_reward = None

        state = self.environment.get_state()
        i = 0
        t0 = time.time()
        while True:
            time_states, percent_in, reward = state

            market_encoding = self.market_encoder.forward(torch.cat(time_states).cpu(), torch.Tensor([percent_in]).cpu(), 'cpu')

            proposed_actions = self.proposer.forward(market_encoding)
            proposed_actions = torch.nn.functional.sigmoid(proposed_actions + torch.randn(1, 2).cpu() * self.sigma)

            replay_proposed = proposed_actions

            policy, value = self.actor_critic.forward(market_encoding, proposed_actions)

            action = int(torch.multinomial(policy, 1))

            if action == 0:
                placed_order = [0, float(proposed_actions[0, 0])]
            elif action == 1:
                placed_order = [1, float(proposed_actions[0, 1])]
            else:
                placed_order = [2]

            replay_mu = policy[0, action]
            replay_place_action = placed_order[0]

            self.environment.step(placed_order)

            # add experience
            replay_initial_time_states = time_states
            replay_initial_percent_in = percent_in

            state = self.environment.get_state()
            if not state:
                break

            time_states, percent_in, reward = state

            replay_final_time_states = time_states
            replay_final_percent_in = percent_in
            replay_reward = reward
            # add experience
            self.experience.append(Experience(replay_initial_time_states,
                                              replay_initial_percent_in,
                                              replay_final_time_states,
                                              replay_final_percent_in,
                                              replay_proposed,
                                              replay_place_action,
                                              replay_mu,
                                              replay_reward))

            i += 1
            if i%100 == 0:
                print(str(i) + "\t" + str(time.time() - t0))
            if i % 1000 == 0:
                break

        self.server.set("experience_" + self.name, pickle.dumps(self.experience))

Experience = namedtuple('Experience',
                        ('initial_time_states',
                         'initial_percent_in',
                         'final_time_states',
                         'final_percent_in',
                         'proposed',
                         'place_action',
                         'mu',
                         'reward'))
