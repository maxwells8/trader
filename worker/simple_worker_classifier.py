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
        self.n_steps_future = int(self.server.get("n_steps_future").decode("utf-8"))

        self.window = networks.WINDOW
        self.start = start

        self.time_states = []
        self.last_time = None

        self.n_sent = 0

    def add_bar(self, bar):
        time_state = [[[bar.open, bar.high, bar.low, bar.close]]]

        if bar.date != self.last_time:
            self.time_states.append(time_state)

        if len(self.time_states) > self.window + self.n_steps_future:
            del self.time_states[0]

        if bar.date != self.last_time and len(self.time_states) == self.window + self.n_steps_future:
            experience = Experience(self.time_states)
            experience = msgpack.packb(experience, use_bin_type=True)
            self.server.lpush("experience", experience)
            # self.server.lpush("experience_dev", experience)
            self.n_sent += 1
            del self.time_states[:self.n_steps_future]
            # self.time_states = []

        self.last_time = bar.date

    def run(self):
        t0 = time.time()
        start = self.start
        while self.n_sent < 100:
            n_seconds = (self.window + self.n_steps_future - len(self.time_states) + (self.n_steps_future * (100 - self.n_sent))) * 60
            # n_seconds = 10000 * 60
            self.zeus.stream_range(start, start + n_seconds, self.add_bar)
            start += n_seconds

        print("time: {time}".format(time=time.time()-t0))

Experience = namedtuple('Experience', ('time_states'))
