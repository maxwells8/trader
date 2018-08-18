import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from networks import *
from environment import *
import redis


class Worker(object):

    def __init__(self, source, window=400):

        self.environment = Env(source, window)
        self.market_encoder = torch.load('models/market_encoder.pt')
        self.actor = torch.load('models/actor.pt')
        self.critic = torch.load('models/critic.pt')
        self.order = torch.load('models/order.pt')

        self.server = redis.Redis("localhost")

        self.experience = []

    def start(self):

        while True:
            time_states, orders, balance, value = self.environment.get_state()

            market_values = []
            for state in time_states:
                market_values.append(torch.from_numpy(state).view(1, 1, state.shape[0]))
