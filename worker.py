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


"""
TODO
-specify which cpu core to use
-fix model usage to fit the proper usage
"""


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

        self.replay = ReplayMemory()

    def run(self):
        value = self.environment.value

        replay_time_states = []
        replay_orders = []
        replay_queried = None
        replay_orders_actions = []
        replay_place_action = None
        replay_reward = 0
        while True:
            state = self.environment.get_state()
            if not state:
                return value

            replay_time_states = time_states

            time_states, open_orders, balance, value, reward = state

            replay_time_states.append(time_states[-1])
            replay_reward = reward
            # add experience
            self.replay.add_experience(Experience(replay_time_states,
                                                  replay_orders,
                                                  replay_queried,
                                                  replay_orders_actions,
                                                  replay_place_action,
                                                  replay_reward))

            market_encoding = self.market_encoder.forward(time_states)

            proposed_actions = self.actor.forward(market_encoding, torch.tensor([balance]), torch.tensor([value]))
            proposed_actions += torch.randn(1, 2) * self.sigma

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
            """
            get this nasty for loop out of here
            """
            for order_i in range(len(open_orders)):
                # THIS WILL NO LONGER WORK -- CHECK THE ORDER NETWORK FOR
                # THE UPDATED VERSION
                advantage = self.order.forward(market_encoding, torch.from_numpy(open_orders[order_i].as_ndarray()))[0]

                action = int(advantage.max(1)[1])
                if np.random.rand() < self.close_epsilon:
                    action = np.random.randint(0, 2)

                if action == 0:
                    replay_orders_actions.append(action)
                elif action == 1:
                    closed_orders.append(order_i)
                    replay_orders_actions.append(action)

            self.environment.step(placed_order, closed_orders)

            if int(self.server.get('update').decode("utf-8")):
                self.replay.push(self.server)
                self.market_encoder = torch.load('models/market_encoder.pt')
                self.actor = torch.load('models/actor.pt')
                self.critic = torch.load('models/critic.pt')
                self.order = torch.load('models/order.pt')


Experience = namedtuple('Experience',
                        ('time_states',
                         'orders',
                         'queried_amount',
                         'orders_actions',
                         'place_action',
                         'reward'))


class ReplayMemory(object):

    def __init__(self):
        self.experiences = []

    def add_experience(self, experience):
        self.experiences.append(experience)

    def push(self, server):

        for experience in self.experiences:
            server.lpush('ER', experience)

        self.experiences = []
