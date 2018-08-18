import pandas as pd
import numpy as np
import time


class Env(object):

    def __init__(self, source, time_window=400):

        self.data = pd.DataFrame(pd.read_csv(source))

        self.time_window = time_window
        self.cur_i = 0

        self.time_states = []
        self.orders = []
        self.balance = 1
        self.value = 1

    def get_state(self):
        if len(self.time_states) == self.time_window:
            del self.time_states[0]

        if self.cur_i == self.data.shape[0]:
            return False

        new_time_state = TimeState(open=self.data['open'][self.cur_i],
                                   high=self.data['high'][self.cur_i],
                                   low=self.data['low'][self.cur_i],
                                   close=self.data['close'][self.cur_i],
                                   time=self.data['time'][self.cur_i],
                                   spread=self.get_rand_spread())

        self.time_states.append(new_time_state)

        self.value = self.balance
        for order in self.orders:
            self.value += order.value(self.time_states[-1].close, self.time_states[-1].spread)

        return self.time_states, self.orders, self.balance, self.value

    def step(self, placed_order, closed_orders):
        if placed_order[0] == 1:
            self.buy(placed_order[1])
        elif placed_order[0] == -1:
            self.sell(placed_order[1])

        for order_i in sorted(closed_orders, reverse=True):
            self.close_order(order_i)

        self.cur_i += 1

    def buy(self, amount):
        new_order = Order(open_time=self.data['time'][self.cur_i + 1],
                          open_price=self.data['close'][self.cur_i] + (self.time_states[-1].spread / 2),
                          quantity=self.balance * amount,
                          buy=True)

        self.orders.append(new_order)

    def sell(self, amount):
        new_order = Order(open_time=self.data['time'][self.cur_i + 1],
                          open_price=self.data['close'][self.cur_i] - (self.time_states[-1].spread / 2),
                          quantity=self.balance * amount,
                          buy=False)

        self.orders.append(new_order)

    def close_order(self, order_i):
        self.balance += self.orders[order_i].value(self.data['close'][self.cur_i], self.time_states[-1].spread)
        del self.orders[order_i]

    @staticmethod
    def get_rand_spread():
        return np.random.gamma(3, 2 / 10000)


class Order(object):

    def __init__(self, open_time, open_price, quantity, buy):

        self.quantity = quantity

        self.buy = buy

        self.open_price = open_price
        self.open_time = open_time

    def as_ndarray(self):
        return np.array([self.open_time, self.open_price, self.quantity, int(self.buy)])

    def value(self, close_price, spread):
        if self.buy:
            return (close_price - (spread / 2)) * self.quantity
        else:
            return -1 * (close_price + (spread / 2)) * self.quantity


class TimeState(object):

    def __init__(self, open, high, low, close, time, spread):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.time = time
        self.spread = spread

    def as_ndarray(self):
        return np.array([self.open, self.high, self.low, self.close, self.time, self.spread])

