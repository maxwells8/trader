import torch
import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
from environment import *
import networks
import pickle
import redis
import math
from zeus.zeus import Zeus

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, instrument, granularity, start, n_steps):

        self.server = redis.Redis("localhost")
        self.zeus = Zeus(instrument, granularity)

        self.window = networks.WINDOW
        self.start = start
        self.n_steps = n_steps

        self.time_states = []
        self.spreads = []

    def add_bar(self, bar):
        time_ = (len(self.time_states) - (self.n_steps / 2)) / (self.n_steps / 2)
        time_state = torch.Tensor([bar.open, bar.high, bar.low, bar.close, time_]).view(1, 1, -1)
        self.time_states.append(time_state)
        self.spreads.append(bar.spread)

    def run(self):
        t0 = time.time()

        while len(self.time_states) != self.n_steps + self.window:
            n_seconds = (self.n_steps + self.window - len(self.time_states)) * 60
            self.zeus.stream_range(self.start, self.start + n_seconds, self.add_bar)
            self.start += n_seconds

        experience = Experience(self.time_states, self.spreads)
        self.server.rpush("experience", pickle.dumps(experience, protocol=pickle.HIGHEST_PROTOCOL))

        print("steps: {steps}, time: {time}".format(steps=self.n_steps, time=time.time()-t0))

Experience = namedtuple('Experience', ('time_states', 'spreads'))
