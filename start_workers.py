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
    n_workers = 24
    server_host = "192.168.0.115"
    server = redis.Redis(server_host)
    server.set("p_new_proposal", 1)
    n_steps = int(server.get("trajectory_steps").decode("utf-8"))
    # instruments = ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"]
    instruments = ["EUR_USD"]
    inst_i = 0

    if server.get("reward_ema") == None:
        server.set("reward_ema", 0)
        server.set("reward_emsd", 0)

    def start_process(name, n):
        global inst_i
        # start = np.random.randint(1136073600, 1548374400)
        start = np.random.randint(1546819200, 1547446980)

        instrument = instruments[inst_i]
        inst_i = (inst_i + 1) % len(instruments)

        process = multiprocessing.Process(target=start_worker, args=(name, instrument, granularity, server_host, start))
        process.start()

        print("starting worker {n}".format(n=n))
        return process

    processes = []
    times = []
    for i in range(n_workers):
        processes.append(start_process(str(i), i))
        times.append(time.time())

    while True:
        for i, process in enumerate(processes):
            while process.is_alive() and time.time() - times[i] < 30:
                time.sleep(0.1)
            if process.is_alive():
                # doing process.terminate() will for whatever reason make it
                # hang. doing process.join(time) doesn't properly close the
                # process, so it uses all the ram and ends up crashing. so this
                # is my janky solution instead of figuring out wtf is going on
                try:
                    print("terminating worker {i}".format(i=i))
                    process.terminate()
                    process.join()
                    inst_i = (inst_i - 1) % len(instruments)
                except WindowsError as e:
                    print("error terminating worker {i}".format(i=i))
                    print(e)

            started = False
            while not started:
                if server.llen("experience") < 1000:
                    processes[i] = start_process(str(i), i)
                    times[i] = time.time()
                    started = True
                else:
                    time.sleep(0.1)
