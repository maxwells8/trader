import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, CuDNNLSTM
from keras.optimizers import Adam

class Trader:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memories = []
        self.rewards = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        #'''
        model.add(CuDNNLSTM(50, input_shape=self.state_shape))
        # model.add(LSTM(10))
        # model.add(Dropout(0.3))
        '''
        model.add(Dense(25, input_shape=self.state_shape, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        #'''
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        element = (state, next_state)
        index = self.memories.index(element)
        if index == -1:
            self.memories.append(element)
            self.rewards.append({})
            self.remember(state, action, reward, next_state)
        else:
            self.rewards[index][action] += reward

    def choose(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # print(np.argmax(act_values[0]))
        return np.argmax(act_values[0])  # returns action

    def best_action(self, reward_dict):
        best = []
        for key in reward_dict.keys():
            reward = reward_dict[key]
            if len(best) == 0 or reward > best[0][-1]:
                best.clear()
            elif reward < best[0][-1]:
                continue
            best.append((key, reward))
        return random.choice(best)

    def replay(self):
        states = []
        targets = []
        for index in range(len(self.memories)):
            state, next_state = self.memories[index]
            action_rewards = self.rewards[index]
            action, reward = self.best_action(action_rewards)
            target = reward
            if index < len(self.memories) - 1:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)[0]
            target_f[action] = target
            targets.append(target_f)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
