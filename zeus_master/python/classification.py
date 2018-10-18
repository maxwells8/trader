#! /usr/bin/python3

from new_trader import Trader
import sys
import numpy as np
from utility import load_bars, load_actions_bars, load_labeled_bars, calc_pips, action_to_str, load_states
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, CuDNNLSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta

np.random.seed(648060351)

EUR_USD = "EUR_USD_M5.json"
GBP_USD = "GBP_USD_M5.json"
EUR_USD_TEST = "EUR_USD_M5_TEST.json"
AUD_CAD = "AUD_CAD_M1.json"
AUD_USD = "AUD_USD_M5.json"
SAMPLE_SIZE = 500
BATCH_SIZE = 64
EPOCHS = 3
MIN_PIPS = 5
MAX_BARS = 10

print("Loading data...")
_, state_bars, states, _ = load_states(EUR_USD, SAMPLE_SIZE, max_bars=20000)
_, test_state_bars, test_states, _ = load_states(GBP_USD, SAMPLE_SIZE)

samples = [first for first, second in states]
sample_bars = state_bars[:-1]

test_states = [first for first, second in test_states]
test_state_bars = test_state_bars[:-1]

# print("{}".format(test_states[0][0]))
# raise "ERROR"

# print("{} - {}".format(samples[0][-1][-1], sample_bars[0][-1]))

print("Labeling Data...")
labels = []
for index in range(len(sample_bars) - MAX_BARS):
    open_price = sample_bars[index][-2]
    max_pips = []
    start = index + 1
    end = index + MAX_BARS
    for b in range(start, end):
        close_price = sample_bars[b][-2]
        temp_pips = [calc_pips(open_price, close_price, action) for action in range(2)]
        temp_pips = [p if p >= 0 else 0 for p in temp_pips]
        if len(max_pips) < len(temp_pips):
            max_pips = temp_pips
        else:
            max_pips = [pip if pip > max_pips[i] else max_pips[i] for i, pip in enumerate(temp_pips)]
    # print(pips)
    max_index = np.argmax(max_pips)
    max_pips = [1 if i == max_index and pips >= MIN_PIPS else 0 for i, pips in enumerate(max_pips)]
    max_pips.append(0 if np.amax(max_pips) == 1 else 1)
    labels.append(np.array(max_pips))

'''
print("Removing Some Wait Actions...")
wait_indices = [index for index, matrix in enumerate(labels) if matrix[2] == 1]
action_indices = [index for index, matrix in enumerate(labels) if matrix[2] == 0]
to_remove = np.random.choice(wait_indices, int(len(wait_indices) - (len(action_indices) * WAIT_RATIO)), replace=False)
samples = np.delete(samples[:len(labels)], to_remove, 0)
labels = np.delete(labels, to_remove, 0)
'''
samples = np.array(samples[:len(labels)])
labels = np.array(labels)


print("{} - {}".format(samples.shape, labels.shape))

print("Training...")
save_file = "save/best_action.hdf5"
checkpoint = ModelCheckpoint(save_file, monitor='val_acc', save_best_only=True, verbose=0, mode='max')

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=samples[0].shape))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(100))
model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(samples, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.33, callbacks=[checkpoint])
model.load_weights(save_file)

print("Testing...")
trades = []
open_action = -1
open_trades = []
len_states = len(test_states) - 1
for index in range(len_states):
    state = test_states[index]
    close_price = test_state_bars[index][-2]
    pred = model.predict(np.array([state]))[0]
    action = np.argmax(pred)
    if action < 2:
        if open_action == action or len(open_trades) == 0:
            open_trades.append((close_price, index))
            open_action = action

    close = []
    for trade_index in range(len(open_trades)):
        open_price, open_index = open_trades[trade_index]
        pips = calc_pips(open_price, close_price, open_action)
        if pips >= MIN_PIPS or index - open_index >= MAX_BARS:
            trades.append(pips)
            close.append(trade_index)
    for c in sorted(close, reverse=True):
        del open_trades[c]

    winnings = [pip for pip in trades if pip > 0]
    losses = [pip for pip in trades if pip <= 0]
    sys.stdout.write(
        "\r[{}/{}] - {:.2f} w/l; {} trades; {} open; {:.2f} pips; {:.2f} unrealized pips; [{:.2f} {:.2f}] avg pips    ".
        format(
            index,
            len_states - 1,
            0 if len(trades) == 0 else len(winnings) / len(trades) * 100.,
            len(trades),
            len(open_trades),
            sum(trades),
            sum([calc_pips(trade[0], close_price, open_action) for trade in open_trades]),
            0 if len(winnings) == 0 else sum(winnings) / len(winnings),
            0 if len(losses) == 0 else sum(losses) / len(losses)))
    sys.stdout.flush()

print("")
