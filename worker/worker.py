import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
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
the worker acts in the environment with a fixed policy for a certain amount of
time, and it also prepares the experience for the optimizer.
"""
class Worker(object):

    def __init__(self, source, name, models_loc, window, n_steps):

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
        self.experience = []

    def run(self):
        reward_ema = 0
        ema_parameter = 0.01
        state = self.environment.get_state()
        initial_time_states, initial_percent_in, _ = state
        t0 = time.time()
        for i_step in range(self.n_steps):

            market_encoding = self.market_encoder.forward(torch.cat(initial_time_states).cpu(), torch.Tensor([initial_percent_in]).cpu(), 'cpu')

            queried_actions = self.proposer.forward(market_encoding, torch.randn(1, 2).cpu() * self.sigma)

            policy, value = self.actor_critic.forward(market_encoding, queried_actions)

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
            reward_ema = ema_parameter * reward + (1 - ema_parameter) * reward_ema
            if len(initial_time_states) == self.window:
                experience = Experience(torch.cat(initial_time_states), initial_percent_in, mu, queried_actions, action, reward, torch.cat(final_time_states), final_percent_in)
                self.experience.append(experience)
                self.server.rpush("experience", pickle.dumps(experience))

            initial_time_states = final_time_states
            initial_percent_in = final_percent_in

            if i_step % 1000 == 0:
                print("{name}\'s ema reward after {steps} steps: {reward}".format(name=self.name, steps=i_step, reward=reward_ema))
                self.market_encoder = torch.load(self.models_loc + '/market_encoder.pt').cpu()
                self.proposer = torch.load(self.models_loc + '/proposer.pt').cpu()
                self.actor_critic = torch.load(self.models_loc + '/actor_critic.pt').cpu()
                self.environment.reset()
                state = self.environment.get_state()
                if not state:
                    break
                initial_time_states, initial_percent_in, _ = state


Experience = namedtuple('Experience', ('initial_time_states',
                                       'initial_percent_in',
                                       'mu',
                                       'proposed_actions',
                                       'place_action',
                                       'reward',
                                       'final_time_states',
                                       'final_percent_in'))
