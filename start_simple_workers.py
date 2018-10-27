import redis
import multiprocessing
import sys
sys.path.insert(0, './worker')
import argparse
import worker
import pandas as pd
import numpy as np
import time
import random
import networks
from start_simple_worker import start_worker

if __name__ == "__main__":

    sources = [
    "./normalized_data/DAT_MT_EURUSD_M1_2010-1.3261691621962404.csv",
    "./normalized_data/DAT_MT_EURUSD_M1_2011-1.3920561137891594.csv",
    "./normalized_data/DAT_MT_EURUSD_M1_2012-1.2854807930908945.csv",
    "./normalized_data/DAT_MT_EURUSD_M1_2013-1.327902744225057.csv",
    "./normalized_data/DAT_MT_EURUSD_M1_2014-1.3285929835705848.csv",
    "./normalized_data/DAT_MT_EURUSD_M1_2015-1.109864962131578.csv",
    "./normalized_data/DAT_MT_EURUSD_M1_2016-1.1071083227321519.csv",
    "./normalized_data/DAT_MT_EURUSD_M1_2017-1.1294884577273274.csv"
    ]
    source_lengths = [len(pd.read_csv(source)) for source in sources]
    spread_func_params = list(np.arange(0, 3, 0.1))
    server = redis.Redis("localhost")
    n_steps = int(server.get("trajectory_steps").decode("utf-8"))

    global n_times
    n_times = 0
    def start_process(name):
        i_source = random.randint(0, 7)
        # i_source = 0
        start = random.randint(0, source_lengths[i_source] - n_steps - networks.WINDOW - 1)
        # start = 0
        process = multiprocessing.Process(target=start_worker, args=(sources[i_source], name, start, n_steps))
        process.start()
        global n_times
        n_times += 1
        print("number of trajectories:", n_times)
        return process

    processes = []
    for i in range(len(spread_func_params)):
        server.set("spread_func_param_" + str(i), spread_func_params[i])
        print("starting worker {worker}: spread param={param}".format(worker=i, param=spread_func_params[i]))
        processes.append(start_process(str(i)))

    while True:
        for i, process in enumerate(processes):
            process.join(10)
            started = False
            while not started:
                if server.llen("experience") < 64:
                    print("starting worker {worker}: spread param={param}".format(worker=i, param=spread_func_params[i]))
                    processes[i] = start_process(str(i))
                    started = True
                else:
                    time.sleep(0.1)
