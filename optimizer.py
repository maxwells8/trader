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

        self.server = redis.Redis("localhost")
        self.gamma = self.server.get("gamma")

        self.experience = []

    def run(self):

        while True:
            # get the inputs to the networks in the right form
            batch = Experience(*zip(*self.experience))  # maybe restructure this so that we aren't using the entire experience each batch
            initial_time_states = torch.cat(batch.time_states[:-1], 1)
            final_time_states = torch.cat(batch.time_states[1:], 1)
            orders = batch.orders # i'm not sure how the hell i'm gonna handle variable orders for each sample of experience
            queried_amount = torch.cat(batch.queried_amount)
            orders_actions = torch.cat(batch.orders_actions)
            place_action = torch.cat(batch.place_action)
            reward = torch.cat(batch.reward)

            # calculate the market_encoding
            initial_market_encoding = self.MEN.forward(initial_time_states)
            final_market_encoding = self.MEN.forward(final_time_states)

            # get expected and actual critic values
            expected = self.CN.forward(initial_market_encoding, queried_amount)
            final_critic_values = self.CN.forward(final_market_encoding, queried_amount)
            actual = reward + (final_critic_values * self.gamma)

            # get the gradients for the actor

            # get the gradients for the critic

            # get the gradients for the order network

            pass
