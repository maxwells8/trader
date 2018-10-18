import sys
import numpy as np
from utility import load_bars, load_actions_bars, load_labeled_bars, calc_pips, action_to_str
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta

np.random.seed(648060351)

EUR_USD = "EUR_USD_M5.json"
GBP_USD = "GBP_USD_M5.json"
EUR_USD_TEST = "EUR_USD_M5_TEST.json"
SAMPLE_SIZE = 100
BATCH_SIZE = 100
WAIT_RATIO = 0.5

REWARD_BAR = 10
MIN_PIPS = 5

x_train, y_train = load_actions_bars(EUR_USD, SAMPLE_SIZE, REWARD_BAR, MIN_PIPS)
test_bars, normalized_bars = load_bars(GBP_USD)


wait_indices = [index for index, matrix in enumerate(y_train) if matrix[2] == 1]
action_indices = [index for index, matrix in enumerate(y_train) if matrix[2] == 0]
to_remove = np.random.choice(wait_indices, int(len(wait_indices) - (len(action_indices) * WAIT_RATIO)), replace=False)
x_train = np.delete(x_train, to_remove, 0)
y_train = np.delete(y_train, to_remove, 0)

model = Sequential()
model.add(LSTM(10, input_shape=tuple(x_train.shape[1:])))
# model.add(LSTM(5, return_sequences=True))
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_hinge', optimizer='rmsprop', metrics=['accuracy'])
# model.load_weights("save/lstm.h5")
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=50, validation_split=0.25, callbacks=[EarlyStopping(patience=10)])
model.save("save/lstm.h5")

print("---\nSimulating...")

pips = 0.0
wins = 0
trades = 0
bars = len(test_bars)
steps = 30
open_action = -1
open_prices = []
for index in range(SAMPLE_SIZE, bars + 1):
    sample = []
    for bar in normalized_bars[index-SAMPLE_SIZE:index]:
        sample.append(bar)

    action = np.argmax(model.predict(np.array([np.array(sample)]))[0])

    try:
        '''
        if action < 2:
            close_price = test_bars[index-1][-2]
            if action != open_action:
                # Close all trades and calc pips
                if len(open_prices) > 0:
                    for price in open_prices:
                        pip = calc_pips(price, close_price, open_action)
                        pips += pip
                        trades += 1
                        if pip > 0.0:
                            wins += 1
                    open_prices.clear()
                else:
                    open_prices.append(close_price)

                open_action = action
            else:
                # Open a new trade
                open_prices.append(close_price)
        '''
        if action < 2:
            current_bar = test_bars[index-1]
            reward_bar = test_bars[index+REWARD_BAR-1]
            pip = calc_pips(current_bar[-2], reward_bar[-2], action)
            pips += pip
            trades += 1
            if pip > 0.0:
                wins += 1
        # '''
    except Exception as e:
        pass
    step = int(index / (bars / steps))
    sys.stdout.write("\r{}/{} [{}] - Action: {}, Positions: {}, Trades: {}, W/L: {:.2f}, Pips: {:.2f}".format(
        index,
        bars,
        ''.join(["=" if step > i or index == bars else "." for i in range(steps)]),
        action_to_str(action).upper(),
        len(open_prices),
        trades,
        (wins / trades) * 100.0 if trades > 0 else 0.0,
        pips))
    sys.stdout.flush()

print()
