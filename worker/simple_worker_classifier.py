import torch
import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
from environment import *
import networks
import msgpack
import redis
import math
from zeus.zeus import Zeus

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, instrument, granularity, start, horizon):
        self.server = redis.Redis("localhost")
        self.zeus = Zeus(instrument, granularity)

        self.window = networks.WINDOW
        self.horizon = horizon
        self.start = start
        self.done = False

        self.time_states = []
        self.spreads = []

        self.steps_between_samples = 180
        self.steps_to_sample = self.window + self.horizon

        self.n_samples_total = 32
        self.n_samples_left = self.n_samples_total

    def add_bar(self, bar):
        self.steps_to_sample -= 1
        time_state = [[[bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]]]

        self.time_states.append(time_state)
        self.spreads.append(bar.spread)

        if len(self.time_states) > self.window + self.horizon:
            del self.time_states[0]
            del self.spreads[0]

        if self.steps_to_sample == 0:
            confidence_interval = np.random.exponential(0.0015)
            final_close = self.time_states[self.window-1][0][0][3]
            gt = (np.array(self.time_states[self.window:])[:, 0, 0, 3] - bar.spread - final_close) / (final_close) > confidence_interval
            lt = (np.array(self.time_states[self.window:])[:, 0, 0, 3] - bar.spread - final_close) / (final_close) < -confidence_interval
            if np.max(gt) == 0 and np.max(lt) == 0:
                output = 2
            elif np.max(gt) == 1 and np.max(lt) == 0:
                output = 0
            elif np.max(gt) == 0 and np.max(lt) == 1:
                output = 1
            elif np.argmax(gt) < np.argmax(lt):
                output = 0
            else:
                output = 1

            experience = Experience(input_time_states=self.time_states[:self.window],
                                    initial_spread=self.spreads[self.window-1],
                                    confidence_interval=confidence_interval,
                                    output=output)

            experience = msgpack.packb(experience, use_bin_type=True)
            # randomly insert the experience
            n_experiences = self.server.llen("experience")
            if n_experiences > 0:
                loc = np.random.randint(0, n_experiences)
                ref = self.server.lindex("experience", loc)
                self.server.linsert("experience", "before", ref, experience)
            else:
                self.server.lpush("experience", experience)

            self.steps_to_sample = self.steps_between_samples
            self.n_samples_left -= 1

        if self.n_samples_left == 0:
            self.done = True

    def run(self):
        t0 = time.time()
        n = 0
        while not self.done:
            n_seconds = self.steps_to_sample * 60
            self.zeus.stream_range(self.start, self.start + n_seconds, self.add_bar)
            self.start += n_seconds
            n += 1
            if n > self.n_samples_total:
                break

        print("time: {time}".format(time=time.time()-t0))

Experience = namedtuple('Experience', (
                                        'input_time_states',
                                        'initial_spread',
                                        'confidence_interval',
                                        'output'
                                    ))
