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


class Worker(object):

    def __init__(self, source, name, window=512):

        self.environment = Env(source, window)
        self.market_encoder = torch.load('models/market_encoder.pt')
        self.actor = torch.load('models/actor.pt')
        self.critic = torch.load('models/critic.pt')
        self.order = torch.load('models/order.pt')

        self.server = redis.Redis("localhost")
        self.name = name.decode("utf-8")
        self.place_epsilon = self.server.get("place_epsilon_" + name)
        self.close_epsilon = self.server.get("close_epsilon_" + name)
        self.sigma = self.server.get("sigma_" + name)

    def run(self):
        value = self.environment.value
        while True:
            state = self.environment.get_state()
            if not state:
                return value
            time_states, open_orders, balance, value, reward = state

            market_values = []
            for state in time_states:
                market_values.append(torch.from_numpy(state.as_ndarray()).float().view(1, 1, state.shape[0]))

            market_encoding = self.market_encoder.forward(market_values)

            proposed_actions = self.actor.forward(market_encoding, torch.tensor([balance]), torch.tensor([value]))
            proposed_actions += torch.randn(1, 2) * self.sigma

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

            closed_orders = []
            for order_i in range(len(open_orders)):
                advantage = self.order.forward(market_encoding, torch.from_numpy(open_orders[order_i].as_ndarray()))[0]

                action = int(advantage.max(1)[1])
                if np.random.rand() < self.close_epsilon:
                    action = np.random.randint(0, 2)

                if action == 1:
                    closed_orders.append(order_i)

            self.environment.step(placed_order, closed_orders)


Experience = namedtuple('Experience',
                        ('time_states',
                         'orders',
                         'queried_amount',
                         'orders_actions',
                         'place_action',
                         'reward',
                         'delta'))


class ReplayMemory(object):

    def __init__(self):
        self.experience = []

    def add_experience(self, experience):
        self.experience.append(experience)

    def push(self, redis_server):
        pass


