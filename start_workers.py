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
from start_worker import start_worker

if __name__ == "__main__":

    sources = [
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2010-1.3261691621962404.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2011-1.3920561137891594.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2012-1.2854807930908945.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2013-1.327902744225057.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2014-1.3285929835705848.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2015-1.109864962131578.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2016-1.1071083227321519.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2017-1.1294884577273274.csv"
    ]
    source_lengths = [len(pd.read_csv(source)) for source in sources]
    proposed_sigmas = [-0.01, 0, 0.01]
    policy_sigmas = [1.1, 1, 0.9]
    spread_func_params = [0, 0, 0]
    models_loc = 'C:\\Users\\Preston\\Programming\\trader\\models'
    window = 128
    n_steps = window + 128
    server = redis.Redis("localhost")

    if server.get("reward_ema") == None:
        server.set("reward_ema", None)
        server.set("reward_emsd", 0)

    global n_times
    n_times = 0
    def start_process(name):
        i_source = random.randint(0, 7)
        i_source = 0
        start = random.randint(0, source_lengths[i_source] - n_steps - 1)
        start = 0
        process = multiprocessing.Process(target=start_worker, args=(sources[i_source], name, models_loc, window, start, n_steps))
        process.start()
        global n_times
        n_times += 1
        print("number of trajectories:", n_times)
        return process

    processes = []
    for i in range(len(proposed_sigmas)):
        server.set("proposed_sigma_" + str(i), proposed_sigmas[i])
        server.set("policy_sigma_" + str(i), policy_sigmas[i])
        server.set("spread_func_param_" + str(i), spread_func_params[i])
        print("starting worker {worker}: proposed sigma={proposed_sigma}, policy sigma={policy_sigma}, spread param={param}".format(worker=i, proposed_sigma=proposed_sigmas[i], policy_sigma=policy_sigmas[i], param=spread_func_params[i]))
        processes.append(start_process(str(i)))

    while True:
        for i, process in enumerate(processes):
            started = False
            while not started:
                process.join()
                if server.llen("experience") < 256:
                    print("starting worker {worker}: proposed sigma={proposed_sigma}, policy sigma={policy_sigma}, spread param={param}".format(worker=i, proposed_sigma=proposed_sigmas[i], policy_sigma=policy_sigmas[i], param=spread_func_params[i]))
                    processes[i] = start_process(str(i))
                    started = True
                else:
                    time.sleep(1)
            print("reward ema:", server.get("reward_ema").decode("utf-8"))
