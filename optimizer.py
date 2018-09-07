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
from worker import Experience
import redis


"""
Keep a target network. Update it using an exponential moving average.
Convert experience to cuda tensors.
"""


class Optimizer(object):

    def __init__(self):
        self.MEN = torch.load('models/market_encoder.pt')
        self.AN = torch.load('models/actor.pt')
        self.CN = torch.load('models/critic.pt')
        self.ON = torch.load('models/order.pt')

        self.MEN_ = torch.load('models/market_encoder.pt')
        self.AN_ = torch.load('models/actor.pt')
        self.CN_ = torch.load('models/critic.pt')
        self.ON_ = torch.load('models/order.pt')


        self.server = redis.Redis("localhost")
        self.gamma = self.server.get("gamma")
        self.tau = self.server.get("tau")

        self.experience = []

    def run(self):

        while True:
            # get the inputs to the networks in the right form
            batch = Experience(*zip(*self.experience))  # maybe restructure this so that we aren't using the entire experience each batch
            initial_time_states = torch.cat(batch.time_states[:-1], 1)
            final_time_states = torch.cat(batch.time_states[1:], 1)
            orders = [orders[i] for orders in batch.orders for i in range(len(orders))]
            queried_amount = torch.cat(batch.queried_amount)
            orders_actions = torch.cat(batch.orders_actions)
            place_action = torch.cat(batch.place_action)
            reward = torch.cat(batch.reward)

            # calculate the market_encoding
            initial_market_encoding = self.MEN.forward(initial_time_states)
            final_market_encoding = self.MEN_.forward(final_time_states)

            # output of actor and critic
            """
            i have an inclination that using CN_ instead of CN in the following
            block of code might help to increase stability. not entirely sure
            about this, but it shouldn't really hurt anything if not.
            """
            proposed_actions = self.AN.forward(initial_market_encoding)
            expected_value = self.CN_.forward(initial_market_encoding, proposed_actions)[1]
            expected_value.backward()

            # get expected and actual critic values
            expected = self.CN.forward(initial_market_encoding, queried_amount)
            final_critic_values = self.CN_.forward(final_market_encoding, queried_amount)
            actual = reward + (final_critic_values * self.gamma)

            # output of orders network
            order_market_encodings = []
            for i, market_encoding in enumerate(initial_market_encoding):
                order_market_encodings.append((market_encoding, len(batch.orders[i])))

            order_actions_ = self.ON(order_market_encodings, orders)
            
