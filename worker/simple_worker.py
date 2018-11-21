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

        self.step = 0

        n_allowed = 8
        self.allowed_steps = np.random.choice([i for i in range(1, self.n_steps + 1)], n_allowed)

    def add_bar(self, bar):
        time_state = torch.Tensor([bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]).view(1, 1, -1)
        if self.step > self.window:
            if self.step - self.window in self.allowed_steps:
                experience = Experience(input_time_states=self.time_states[:self.window],
                                        initial_spread=self.spreads[self.window-1],
                                        final_time_state=time_state,
                                        final_spread=bar.spread,
                                        steps=self.step - self.window)

                experience = pickle.dumps(experience, protocol=pickle.HIGHEST_PROTOCOL)
                # randomly insert the experience
                n_experiences = self.server.llen("experience")
                if n_experiences > 0:
                    loc = np.random.randint(0, n_experiences)
                    ref = self.server.lindex("experience", loc)
                    self.server.linsert("experience", "before", ref, experience)
                else:
                    self.server.lpush("experience", experience)

        else:
            self.time_states.append(time_state)
            self.spreads.append(bar.spread)

        self.step += 1

    def run(self):
        t0 = time.time()
        n = 0
        while self.step != self.n_steps + self.window:
            n_seconds = (self.n_steps + self.window - len(self.time_states)) * 60
            self.zeus.stream_range(self.start, self.start + n_seconds, self.add_bar)
            self.start += n_seconds
            n += 1

        print("steps: {steps}, time: {time}".format(steps=self.n_steps, time=time.time()-t0))

Experience = namedtuple('Experience', ('input_time_states', 'initial_spread', 'final_time_state', 'final_spread', 'steps'))
