import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import redis
import multiprocessing
import sys
sys.path.insert(0, './worker')
sys.path.insert(0, './optimizer')
from normalize_data import normalize_and_save
from networks import *
from environment import *
import worker
from start_worker import start_worker
import optimizer


if __name__ == "__main__":
    server = redis.Redis("localhost")
    ##### data_loc = normalize_and_save("data/DAT_MT_EURUSD_M1_2017.csv", True)

    # source = "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2017-1.1294884577273274.csv"
    # name = '0'
    # models_loc = './models'
    # window = 256
    # n_steps = 1440
    #
    # server.set("sigma_" + name, 0.05)
    #
    # p = multiprocessing.Process(target=start_worker, args=(source, name, models_loc, window, n_steps))
    # p.start()

    server.set("gamma", 0.99)
    server.set("optimizer_tau", 0.05)
    server.set("optimizer_max_rho", 1)

    server.set("optimizer_proposed_weight", 2/9)
    server.set("optimizer_critic_weight", 2/9)
    server.set("optimizer_actor_weight", 2/9)
    server.set("optimizer_entropy_weight", 2/9)
    server.set("optimizer_weight_penalty", 1/9)

    server.set("optimizer_batch_size", 4)

    this_optimizer = optimizer.Optimizer('models')
    this_optimizer.run()
