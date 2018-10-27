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
            for time_state in time_states:
                for time_state_ in time_state.squeeze():
                    try:
                        float(time_state_)
                        assert not np.isnan(time_state_)
                    except Exception:
                        raise ValueError("Incorrect value in time state: " + str(time_state_))

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

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    server = redis.Redis("localhost")
    server.set("spread_func_param_test", 0)
    source = "../normalized_data/DAT_MT_EURUSD_M1_2010-1.3261691621962404.csv"
    start = np.random.randint(0, 200000)
    # start = 0
    n_steps = int(server.get("trajectory_steps").decode("utf-8"))
    test = True
    sum_rewards = 0
    worker = Worker(source, "test", start, n_steps)
    worker.run()
