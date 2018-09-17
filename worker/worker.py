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

        self.environment = Env(source, window)
        self.market_encoder = torch.load(models_loc + '/market_encoder.pt').cpu()
        self.market_encoder.device = 'cpu'
        self.actor = torch.load(models_loc + '/actor.pt').cpu()
        self.critic = torch.load(models_loc + '/critic.pt').cpu()
        self.order = torch.load(models_loc + '/order.pt').cpu()

        self.server = redis.Redis("localhost")
        self.name = name
        self.place_epsilon = float(self.server.get("place_epsilon_" + name).decode("utf-8"))
        self.close_epsilon = float(self.server.get("close_epsilon_" + name).decode("utf-8"))
        self.sigma = float(self.server.get("sigma_" + name).decode("utf-8"))

        self.replay = ReplayMemory()

    def run(self):
        value = self.environment.value

        replay_time_states = []
        replay_orders = []
        replay_queried = None
        replay_orders_actions = []
        replay_place_action = None
        replay_reward = 0
        replay_orders_rewards = []

        time_states = []
        i = 0
        while True:
            print(i)
            state = self.environment.get_state()
            if not state:
                return value

            replay_time_states = time_states

            time_states, open_orders, balance, value, reward, orders_rewards = state

            replay_time_states.append(time_states[-1])
            replay_reward = reward
            replay_orders_rewards = orders_rewards
            # add experience
            self.replay.add_experience(Experience(replay_time_states,
                                                  replay_orders,
                                                  replay_queried,
                                                  replay_orders_actions,
                                                  replay_place_action,
                                                  replay_reward,
                                                  replay_orders_rewards))

            market_encoding = self.market_encoder.forward(torch.cat(time_states).cpu())

            proposed_actions = self.actor.forward(market_encoding, torch.tensor([balance]).cpu(), torch.tensor([value]).cpu())
            proposed_actions += torch.randn(1, 2).cpu() * self.sigma

            replay_queried = proposed_actions

            # buy, sell, or neither
            Q_actions = self.critic.forward(market_encoding, proposed_actions)

            if np.random.rand() < self.place_epsilon:
                action = np.random.randint(0, 3)
            else:
                action = int(Q_actions.max(1)[1])

            if action == 0:
                placed_order = [0, float(proposed_actions[0, 0])]
            elif action == 1:
                placed_order = [1, float(proposed_actions[0, 1])]
            else:
                placed_order = [2]

            replay_place_action = placed_order
            replay_orders = open_orders

            replay_orders_actions = []
            closed_orders = []
            # go through each order
            advantage, _ = self.order.forward([(market_encoding, int(len(open_orders)))], open_orders)
            actions = advantage.max(1)[1]
            for i, action in enumerate(actions):
                if np.random.rand() < self.close_epsilon:
                    action = np.random.randint(0, 2)

                if action == 1:
                    closed_orders.append(open_orders[i])
                replay_orders_actions.append(action)

            self.environment.step(placed_order, closed_orders)

            if int(self.server.get('worker_update').decode("utf-8")):
                self.replay.push(self.server)
                self.market_encoder = torch.load(models_loc + '/market_encoder.pt').cpu()
                self.actor = torch.load(models_loc + '/actor.pt').cpu()
                self.critic = torch.load(models_loc + '/critic.pt').cpu()
                self.order = torch.load(models_loc + '/order.pt').cpu()

            i += 1


Experience = namedtuple('Experience',
                        ('time_states',
                         'orders',
                         'queried_amount',
                         'orders_actions',
                         'place_action',
                         'reward',
                         'orders_rewards'))


class ReplayMemory(object):

    def __init__(self):
        self.experiences = []

    def add_experience(self, experience):
        self.experiences.append(experience)

    def push(self, server, worker_name):
        server.set(worker_name, pickle.dumps(self.experiences))

        self.experiences = []
