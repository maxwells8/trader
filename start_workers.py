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
from start_worker import start_worker

if __name__ == "__main__":

    granularity = "M1"
    n_workers = 16
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_loc = dir_path + '/models/'
    server = redis.Redis("localhost")
    n_steps = int(server.get("trajectory_steps").decode("utf-8"))

    if server.get("reward_ema") == None:
        server.set("reward_ema", 0)
        server.set("reward_emsd", 0)

    n_times = 0
    def start_process(name):
        global n_times
        instrument = np.random.choice(["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"])
        start = np.random.randint(1136073600, 1546300800)
        # instrument = "EUR_USD"
        # start = np.random.randint(1546214400, 1546819200)

        process = multiprocessing.Process(target=start_worker, args=(name, instrument, granularity, models_loc, start))
        process.start()

        n_times += 1
        print("n: {n}".format(n=n_times))
        return process

    processes = []
    times = []
    for i in range(n_workers):
        processes.append(start_process(str(i)))
        times.append(time.time())

    while True:
        for i, process in enumerate(processes):
            while process.is_alive() and time.time() - times[i] < 40:
                time.sleep(0.1)
            if process.is_alive():
                # doing process.terminate() will for whatever reason make it
                # hang. doing process.join(time) doesn't properly close the
                # process, so it uses all the ram and ends up crashing. so i
                # need to just get at the root of it and figure out what's
                # hanging in the worker.
                try:
                    print("terminating process")
                    process.terminate()
                    process.join()
                except WindowsError as e:
                    print("error terminating process:")
                    print(e)

            started = False
            while not started:
                if server.llen("experience") < 1024:
                    print("starting worker")
                    processes[i] = start_process(str(i))
                    times[i] = time.time()
                    started = True
                else:
                    time.sleep(0.1)
