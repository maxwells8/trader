import torch
import pandas as pd
import numpy as np
import time


"""
add functionality to get some random price within the high and low of the
current price. this can be used to continually check whether a trade is worth
it, not just at the very beginning of the tick.
"""
class Env(object):

    def __init__(self, source, spread_func=(lambda _:np.random.gamma(3, 2 / 10000))(None), time_window=512):
        self.data = pd.DataFrame(pd.read_csv(source))

        self.time_window = time_window
        self.cur_i = 0

        self.time_states = []
        self.orders = []
        self.balance = 1
        self.prev_value = 1
        self.value = 1

        self.spread_func = spread_func

        first_time_state = TimeState(open=self.data['open'][self.cur_i],
                                     high=self.data['high'][self.cur_i],
                                     low=self.data['low'][self.cur_i],
                                     close=self.data['close'][self.cur_i],
                                     time=self.data['time'][self.cur_i],
                                     spread=self.spread_func,
                                     balance=self.balance,
                                     value=self.value)

        self.time_states.append(first_time_state.as_tensor())

    def get_state(self):

        if self.cur_i == self.data.shape[0]:
            return False

        return self.time_states, (self.value - self.balance) / self.value, self.reward()

    def step(self, placed_order):
        if placed_order[0] == 0:
            self.buy(placed_order[1] * self.balance)
        elif placed_order[0] == 1:
            self.sell(placed_order[1] * self.balance)

        self.cur_i += 1

        if self.cur_i == self.data.shape[0]:
            return False

        if len(self.time_states) == self.time_window:
            del self.time_states[0]

        self.prev_value = self.value
        self.calculate_value()

        new_time_state = TimeState(open=self.data['open'][self.cur_i],
                                   high=self.data['high'][self.cur_i],
                                   low=self.data['low'][self.cur_i],
                                   close=self.data['close'][self.cur_i],
                                   time=self.data['time'][self.cur_i],
                                   spread=self.spread_func(None),
                                   balance=self.balance,
                                   value=self.value)

        self.time_states.append(new_time_state.as_tensor())

    def buy(self, amount):
        new_order = Order(open_time=self.data['time'][self.cur_i + 1],
                          open_price=self.data['close'][self.cur_i] + (self.time_states[-1].spread / 2),
                          quantity=amount,
                          buy=True)

        self.orders.append(new_order)

    def sell(self, amount):
        new_order = Order(open_time=self.data['time'][self.cur_i + 1],
                          open_price=self.data['close'][self.cur_i] - (self.time_states[-1].spread / 2),
                          quantity=amount,
                          buy=False)

        self.orders.append(new_order)

    def close_order(self, order_i):
        self.balance += self.orders[order_i].value(self.data['close'][self.cur_i], self.time_states[-1].spread)
        del self.orders[order_i]

    def calculate_value(self):
        self.value = self.balance
        for order in self.orders:
            self.value += order.value(self.time_states[-1].close, self.time_states[-1].spread)

        return self.value

    def reward(self):
        self.calculate_value()
        return self.value - self.prev_value

    def orders_rewards(self):
        rewards = []
        for order in self.orders:
            rewards.append(self.time_states[-1].close - order.open_price)

        return rewards


class Order(object):

    def __init__(self, open_time, open_price, quantity, buy):

        self.quantity = quantity

        self.open_price = open_price
        self.open_time = open_time

        self.buy = buy

    def as_ndarray(self):
        return np.array([self.open_time, self.open_price, self.quantity, int(self.buy)])

    def value(self, close_price, spread):
        if self.buy:
            return (close_price - (spread / 2)) * self.quantity
        else:
            return -1 * (close_price + (spread / 2)) * self.quantity


class TimeState(object):

    def __init__(self, open, high, low, close, time, spread, balance, value):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.time = time
        self.spread = spread
        self.balance = balance
        self.value = value

    def as_ndarray(self):
        return np.array([self.open,
                         self.high,
                         self.low,
                         self.close,
                         self.time,
                         self.spread,
                         self.balance,
                         self.value])

    def as_tensor(self):
        return torch.Tensor([self.open,
                             self.high,
                             self.low,
                             self.close,
                             self.time,
                             self.spread,
                             self.balance,
                             self.value]).float().view(1, 1, -1)
