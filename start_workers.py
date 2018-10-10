import redis
import multiprocessing
import sys
sys.path.insert(0, './worker')
import argparse
import worker
import pandas as pd
import time
import random
from start_worker import start_worker

if __name__ == "__main__":
    sources = [
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2014-1.3285929835705848.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2015-1.109864962131578.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2016-1.1071083227321519.csv",
    "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2017-1.1294884577273274.csv"
    ]
    source_lengths = [len(pd.read_csv(source)) for source in sources]
    sigmas = [0.01, 0.025, 0.05, 0.1]
    models_loc = 'C:\\Users\\Preston\\Programming\\trader\\models'
    window = 256
    n_steps = 1440
    server = redis.Redis("localhost")
    server.set("sigma_0", 0.01)

    def start_process(name):
        i_source = random.randint(0, 3)
        start = random.randint(0, source_lengths[i_source] - n_steps - 1)
        process = multiprocessing.Process(target=start_worker, args=(sources[i_source], name, models_loc, window, start, n_steps))
        process.start()
        return process

    processes = []
    for i in range(4):
        server.set("sigma_" + str(i), sigmas[i])
        processes.append(start_process(str(i)))

    while True:
        for i, process in enumerate(processes):
            if not process.is_alive():
                processes[i] = start_process(str(i))
        time.sleep(0.1)
