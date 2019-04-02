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
    n_workers = 1
    server_host = "192.168.0.115"
    server = redis.Redis(server_host)
    n_steps = int(server.get("trajectory_steps").decode("utf-8"))
    # instruments = ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"]
    instruments = ["EUR_USD"]
    inst_i = 0

    for inst in instruments:
        server.set("test_ema_" + inst, 0)
        server.set("test_emsd_" + inst, 0)

    reward_tau = 0.001
    server.set("test_reward_tau", reward_tau)
    server.set("test_reward_ema", 0)
    server.set("test_reward_emsd", 0)

    def start_process(name):
        name = "test" + name
        global n_times
        global inst_i
        start = np.random.randint(1136073600, 1548374400)
        # start = np.random.randint(1546819200, 1546819200 + (1440 * 60 * 4))
        # start = 1546819200
        # start = 1546819200 + (1440 * 60 * 4)
        instrument = instruments[inst_i]
        inst_i = (inst_i + 1) % len(instruments)

        process = multiprocessing.Process(target=start_worker, args=(name, instrument, granularity, server_host, start, True))
        process.start()

        return process

    processes = []
    times = []
    for i in range(n_workers):
        processes.append(start_process(str(i)))
        times.append(time.time())

    while True:
        for i, process in enumerate(processes):
            while process.is_alive() and time.time() - times[i] < 300:
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
                    print("joining process")
                    process.join(5)
                    inst_i = (inst_i - 1) % len(instruments)
                except WindowsError as e:
                    print("error terminating process:")
                    print(e)

            print("starting test worker")
            processes[i] = start_process(str(i))
            times[i] = time.time()
