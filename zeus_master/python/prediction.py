#! /usr/bin/python3

from new_trader import Trader
import sys
import numpy as np
from utility import load_bars, load_actions_bars, load_labeled_bars, calc_pips, action_to_str, load_states
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv2D, MaxPooling2D, CuDNNLSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta

np.random.seed(648060351)

EUR_USD = "EUR_USD_M5.json"
GBP_USD = "GBP_USD_M5.json"
EUR_USD_TEST = "EUR_USD_M5_TEST.json"
SAMPLE_SIZE = 500
BATCH_SIZE = 150
EPOCHS = 50
MIN_PIPS = 10
MAX_BARS = 10

print("Loading data...")
_, state_bars, states, _ = load_states(EUR_USD, SAMPLE_SIZE)
_, additional_state_bars, additional_states, _ = load_states(EUR_USD_TEST, SAMPLE_SIZE)
_, test_state_bars, test_states, _ = load_states(EUR_USD_TEST, SAMPLE_SIZE)

test_states = [first for first, second in test_states]
test_state_bars = test_state_bars[:-1]
samples = [[first for first, second in states], [first for first, second in additional_states]]
sample_bars = [state_bars[:-1], additional_state_bars[:-1]]

# print("{} - {}".format(samples[0][-1][-1], sample_bars[0][-1]))

print("Labeling data...")
for sample_bar in sample_bars:
