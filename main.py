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
    # data_loc = normalize_and_save("data/DAT_MT_EURUSD_M1_2017.csv", True)
    source = "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2017-1.1294884577273274.csv"
    name = '0'
    models_loc = './models'
    window = 512

    p = multiprocessing.Process(target=start_worker, args=(source, name, models_loc, window))
    p.start()
