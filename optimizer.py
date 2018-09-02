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
Keep a target network. Update it using an exponential moving average.
Convert experience to cuda tensors.
"""

Experience = namedtuple('Experience',
                        ('time_states',
                         'orders',
                         'queried_amount',
                         'orders_actions',
                         'place_action',
                         'reward'))


class Optimizer(object):

    def __init__(self):
        self.MEN = torch.load('models/market_encoder.pt')
        self.AN = torch.load('models/actor.pt')
        self.CN = torch.load('models/critic.pt')
        self.ON = torch.load('models/order.pt')

        self.target_MEN = self.MEN
        self.AN = self.AN
        self.CN = self.CN
        self.ON = self.ON

        self.server = redis.Redis("localhost")

        self.experience = []

    def run(self):
        while True:

            market_encoding = self.MEN.forward()
