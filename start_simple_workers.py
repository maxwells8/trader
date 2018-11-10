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

    server = redis.Redis("localhost")
    n_steps = int(server.get("trajectory_steps").decode("utf-8"))
    n_workers = 32

    n_times = 0
    def start_process():
        global n_times
        instrument = np.random.choice(["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"])
        granularity = "M1"

        # pick a time since 2006, because the volume is too small earlier
        start = np.random.randint(1136073600, int(time.time()) - (60 * (n_steps + networks.WINDOW)))
        # since 7 nov 2018 for testing
        # start = np.random.randint(1541548800, int(time.time()) - (60 * (n_steps + networks.WINDOW)))

        print("starting worker {worker}... instrument: {instrument}, granularity: {gran}, start: {start}".format(worker=n_times,instrument=instrument, gran=granularity, start=start))

        process = multiprocessing.Process(target=start_worker, args=(instrument, granularity, start, n_steps))
        process.start()
        n_times += 1
        return process

    processes = []
    for i in range(n_workers):
        processes.append(start_process())

    while True:
        for i, process in enumerate(processes):
            process.join(30)
            started = False
            while not started:
                if server.llen("experience") < 256:
                    processes[i] = start_process()
                    started = True
                else:
                    time.sleep(0.1)
