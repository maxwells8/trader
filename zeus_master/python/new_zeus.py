#! /usr/bin/python3

from new_trader import Trader
import sys
import numpy as np
from utility import load_bars, load_actions_bars, load_labeled_bars, calc_pips, action_to_str, load_states
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta

np.random.seed(648060351)

EUR_USD = "EUR_USD_M5.json"
GBP_USD = "GBP_USD_M5.json"
EUR_USD_TEST = "EUR_USD_M5_TEST.json"
SAMPLE_SIZE = 500
BATCH_SIZE = 2000
WAIT_RATIO = 0.5
EPOCHS = 200
MIN_PIPS = 10
MAX_BARS = 10
PROGRESS_LENGTH = 30

print("Loading data...")
_, _, states, state_bars = load_states(EUR_USD, SAMPLE_SIZE)
_, _, additional_states, additional_state_bars = load_states(EUR_USD_TEST, SAMPLE_SIZE)

samples = [states, additional_states]
sample_bars = [state_bars, additional_state_bars]

print("Training...")
agent = Trader(states[0][0].shape, 3)
num_states = len(samples[0]) + len(samples[1]) - 1
best_pips = 0.0
best_wl = 50.0
for epoch in range(EPOCHS):
    trades = 0
    wins = 0
    pips = 0.0
    rewards = 0
    state_index = 0
    print("\nEpoch {}/{}".format(epoch, EPOCHS))
    for sample in samples:
        open_action = 2
        open_prices = []
        for index in range(len(sample)-MAX_BARS):
            current_state, next_state = sample[index]
            current_state = np.array([current_state])
            next_state = np.array([next_state])
            close_price = current_state[0][-1][-2]
            action = agent.choose(current_state)
            reward_bars = sample_bars[index:index+MAX_BARS]
            reward = MIN_PIPS
            if action < 2:
                for bar in reward_bars:
                    pip = calc_pips(close_price, bar[-2], action)
            '''
            done = False

            if action < 2:
                if open_action == 2:
                    open_action = action
                    open_prices.append(close_price)
                elif open_action != action:
                    done = True
                    position_pips = 0.0
                    for open_price in open_prices:
                        pip = calc_pips(open_price, close_price, open_action)
                        position_pips += pip
                        trades += 1
                        if pip > 0:
                            wins += 1
                    pips += position_pips
                    reward = 1 if position_pips > MIN_PIPS else -1
                    rewards += reward
            '''

            # Print
            wl = ((wins / trades) * 100 if trades > 0 else 0)
            progress = int(state_index / (num_states / PROGRESS_LENGTH))
            sys.stdout.write("\r{}/{} [{}] - Epsilon: {:.2f}, Action: {}, Trades: {}, Pips: {:.2f}, W/L: {:.2f}, Reward: {}".format(
                state_index,
                num_states,
                ''.join(["=" if progress > i or state_index == num_states else "." for i in range(PROGRESS_LENGTH)]),
                agent.epsilon,
                action_to_str(open_action).upper(),
                trades,
                pips,
                wl,
                rewards))
            sys.stdout.flush()

            agent.remember(current_state, action, reward, next_state, done)

            if done:
                open_action = 2
                open_prices = []
            
            state_index += 1
    
    if agent.epsilon < 0.5 and pips > best_pips and wl > best_wl:
        sys.stdout.write("\nSaving Weights.")
        sys.stdout.flush()

        best_pips = pips
        best_wl = wl
        agent.save("save/zeus-dqn-ema.h5")

    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)

    if epoch > 25:
        agent.decay()



'''

    Things to look into:
        - More Indicators:
            - Bands
            - Stochastic
            - EMA
        - Reward function
        - More bars to sample from
        - Incorporate multiple time periods (M1, M5, M15, M30, etc.)
        - States?
        - Early Stopping?
        - Best weights

'''
