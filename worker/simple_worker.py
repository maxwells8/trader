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

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, source, name, start, n_steps):

        self.server = redis.Redis("localhost")
        self.name = name
        self.spread_func_param = float(self.server.get("spread_func_param_" + name).decode("utf-8"))

        self.window = networks.WINDOW
        self.n_steps = n_steps

        self.environment = Env(source, start, n_steps, self.spread_func_param, self.window, get_time=True)

    def run(self):
        all_rewards = 0

        time_states = []
        spreads = []

        state = self.environment.get_state()
        time_states_, _, spread_, _ = state
        time_states = time_states_
        t0 = time.time()
        for i_step in range(self.n_steps):

            spreads.append(spread_)

            self.environment.step([4])

            state = self.environment.get_state()
            if not state:
                break

            time_states_, _, spread_, _ = state

            time_states.append(time_states_[-1])

        experience = Experience(time_states, spreads)
        self.server.rpush("experience", pickle.dumps(experience, protocol=pickle.HIGHEST_PROTOCOL))

        print("name: {name}, steps: {steps}, time: {time}".format(name=self.name, steps=i_step, time=time.time()-t0))

Experience = namedtuple('Experience', ('time_states', 'spreads'))
