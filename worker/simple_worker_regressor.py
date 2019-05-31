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

    def __init__(self, instrument, granularity, server_host, start, max_time):
        self.server = redis.Redis(server_host)
        self.zeus = Zeus(instrument, granularity)

        self.window = networks.WINDOW + 10
        self.start = start

        self.time_states = []
        self.last_time = None
        self.max_time = max_time

        self.max_sent = 100
        self.n_sent = 0

        self.t0 = 0

    def add_bar(self, bar):
        if self.n_sent == self.max_sent or bar.date > self.max_time:
            print("experiences sent: {n_sent}, time: {time}".format(n_sent=self.n_sent, time=time.time()-self.t0))
            quit()
        time_state = [bar.open, bar.high, bar.low, bar.close]

        if bar.date != self.last_time:
            self.time_states.append(time_state)
            self.last_time = bar.date

            if len(self.time_states) > self.window:
                del self.time_states[0]

            if len(self.time_states) == self.window:
                experience = Experience(time_states=self.time_states)
                experience = msgpack.packb(experience, use_bin_type=True)
                n_experiences = self.server.llen("experience")
                try:
                    loc = np.random.randint(0, n_experiences)
                    ref = self.server.lindex("experience", loc)
                    self.server.linsert("experience", "before", ref, experience)
                except Exception as e:
                    self.server.lpush("experience", experience)
                # self.server.lpush("experience", experience)
                self.n_sent += 1

                # self.time_states = []
                del self.time_states[0]

    def run(self):
        self.t0 = time.time()
        start = self.start
        while self.n_sent < self.max_sent:
            n_seconds = (self.window - len(self.time_states) + (self.max_sent - self.n_sent)) * 60 * 5
            self.zeus.stream_range(start, start + n_seconds, self.add_bar)
            start += n_seconds

        print("experiences sent: {n_sent}, time: {time}".format(n_sent=self.n_sent, time=time.time()-self.t0))

Experience = namedtuple('Experience', ('time_states'))
