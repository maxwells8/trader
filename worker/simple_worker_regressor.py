import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
import networks
import msgpack
import redis
from zeus.zeus import Zeus

class Worker(object):

    def __init__(self, instrument, granularity, server_host, start):
        self.server = redis.Redis(server_host)
        self.zeus = Zeus(instrument, granularity)

        self.window = networks.WINDOW + 30
        self.start = start

        self.time_states = []
        self.last_time = None

        self.n_sent = 0

    def add_bar(self, bar):
        time_state = [[[bar.open, bar.high, bar.low, bar.close]]]

        if bar.date != self.last_time:
            self.time_states.append(time_state)
            self.last_time = bar.date

            if len(self.time_states) > self.window:
                del self.time_states[0]

            if len(self.time_states) == self.window:
                experience = Experience(time_states=self.time_states)
                experience = msgpack.packb(experience, use_bin_type=True)
                self.server.lpush("experience", experience)
                # self.server.lpush("experience_dev", experience)
                self.n_sent += 1

                del self.time_states[:30]

    def run(self):
        t0 = time.time()
        start = self.start
        n_seconds = (self.window - len(self.time_states) + (100 - self.n_sent)) * 60 * 100
        self.zeus.stream_range(start, start + n_seconds, self.add_bar)
        start += n_seconds

        print(self.n_sent)
        print("time: {time}".format(time=time.time()-t0))

Experience = namedtuple('Experience', ('time_states'))
