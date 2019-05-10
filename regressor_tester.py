import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
import torch
import networks
from networks import *
import msgpack
import redis
import copy
from zeus.zeus import Zeus
import matplotlib.pyplot as plt

class Worker(object):

    def __init__(self, instrument, granularity, server_host, start, models_loc):
        self.server = redis.Redis(server_host)
        self.zeus = Zeus(instrument, granularity, margin=1)
        self.models_loc = models_loc

        self.window = networks.WINDOW
        self.start = start

        self.time_states = []
        self.last_time = None

        self.generator = Generator()
        try:
            self.generator.load_state_dict(torch.load(self.models_loc + 'generator.pt', map_location='cpu'))
        except FileNotFoundError as e:
            print("model load failed", e)
            self.generator = Generator()

        self.generator.eval()

        self.n_future_samples = 1
        self.n_steps_future = 10

        self.step = 0

        self.last_price = None
        self.last_action = None
        self.steps_since_trade = 0
        self.n_correct = 0
        self.n_trades = 1

    def add_bar(self, bar):
        time_state = [[[bar.open, bar.high, bar.low, bar.close]]]

        if bar.date != self.last_time:
            self.time_states.append(time_state)

        if len(self.time_states) > self.window + self.n_steps_future:
            del self.time_states[0]

            if bar.date != self.last_time:
                input_time_states = []
                for time_state in self.time_states:
                    input_time_states.append(torch.Tensor(time_state).view(1, 1, networks.D_BAR))
                time_state_values = torch.cat(input_time_states, dim=1)[0, :, 3].detach().numpy()
                # plt.plot(np.arange(0, self.window + self.n_steps_future), time_state_values, color='b')

                input_time_states_original = []
                for time_state in self.time_states[:self.window]:
                    input_time_states_original.append(torch.Tensor(time_state).view(1, 1, networks.D_BAR))

                future_samples = []
                for i_sample in range(self.n_future_samples):
                    input_time_states = input_time_states_original
                    future_samples_ = []

                    for i_future in range(self.n_steps_future):
                        input_time_states_ = torch.cat(input_time_states[len(future_samples_):] + future_samples_, dim=1).clone()
                        mean = input_time_states_.contiguous().view(1, -1).mean(1).view(1, 1, 1)
                        std = input_time_states_.contiguous().view(1, -1).std(1).view(1, 1, 1)
                        input_time_states_ = (input_time_states_ - mean) / std
                        input_time_states_ = input_time_states_.transpose(0, 1)

                        _, encoding_mean, encoding_std = self.encoder(input_time_states_)
                        sample_encoding = torch.normal(encoding_mean, encoding_std * 0)
                        generation = self.generator(sample_encoding)
                        next_bar = generation + input_time_states_[-1, 0, 3].repeat(4, 1).transpose(0, 1)
                        next_bar = next_bar * std + mean
                        future_samples_.append(next_bar.view(1, 1, networks.D_BAR))

                    future_samples.append(torch.cat(future_samples_, dim=1))
                    time_state_values = torch.cat([input_time_states[-1]] + future_samples_, dim=1)[0, :, 3].detach().numpy()
                #     plt.plot(np.arange(self.window - 1, self.window + self.n_steps_future), time_state_values, color='g', alpha=0.1)
                # plt.show()
                predicted_future = torch.cat(future_samples, dim=0)[:, -1, 3].mean().item()
                actual_future = self.time_states[-1][0][0][3]
                original = input_time_states[-1][0, 0, 3].item()
                print(round(actual_future - original, 7), round(predicted_future - original, 7))
                if round((actual_future - original) / (abs(actual_future - original) + 1e-9), 5) == round((predicted_future - original) / (abs(predicted_future - original) + 1e-9), 5):
                    print("correct direction")
                else:
                    print("incorrect direction")
                print()
                self.time_states = []


                # action = prediction[0].argmax().item()
                #
                # in_ = self.zeus.position_size()
                # available_ = self.zeus.units_available()
                # percent_in = (in_ / (abs(in_) + available_ + 1e-9))
                # if action == 0:
                #     desired_percent = -self.tradeable_percentage
                # elif action == 1:
                #     desired_percent = 0
                # elif action == 2:
                #     desired_percent = self.tradeable_percentage
                # # desired_percent = self.tradeable_percentage * action + -self.tradeable_percentage * (1 - action)
                # if desired_percent == 0:
                #     self.zeus.close_units(self.zeus.position_size())
                # elif desired_percent > percent_in:
                #     if percent_in < 0:
                #         self.zeus.close_units(self.zeus.position_size())
                #     total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                #     self.zeus.place_trade(int(desired_percent * total_tradeable), "Long")
                # elif desired_percent < percent_in:
                #     if percent_in > 0:
                #         self.zeus.close_units(self.zeus.position_size())
                #     total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                #     self.zeus.place_trade(int(-desired_percent * total_tradeable), "Short")
                #
                # self.steps_since_trade = 0
                #
                # if self.last_price is not None:
                #     print('-------------------------------------------------------')
                #     if (bar.close > self.last_price and self.last_action == 2) or (bar.close < self.last_price and self.last_action == 0):
                #         self.n_correct += 1
                #     if action in [0, 2]:
                #         self.n_trades += 1
                #
                #     print(bar.close, self.last_price, self.last_action)
                #     print(prediction.detach().numpy(), action, self.n_correct / self.n_trades)
                #     print('-------------------------------------------------------')
                #
                # self.last_price = copy.deepcopy(bar.close)
                # self.last_action = action

        # print("days since start: {elapsed}, steps: {steps}, profit: {profit}".format(
        #     elapsed=(bar.date-self.start) / 86400,
        #     steps=self.step,
        #     profit=self.zeus.unrealized_balance()
        # ))
        self.last_time = bar.date
        self.step += 1
        self.steps_since_trade += 1

    def run(self):
        t0 = time.time()
        with torch.no_grad():
            start = self.start
            while True:
                n_seconds = 60 * 1440 * 30
                self.zeus.stream_range(start, start + n_seconds, self.add_bar)
                start += n_seconds

        print("time: {time}".format(time=time.time()-t0))


if __name__ == "__main__":
    """
    1/1/06 -> 1136073600
    1/1/12 -> 1325376000
    1/1/17 -> 1483228800
    1/1/18 -> 1514764800
    1/1/19 -> 1546300800
    """
    worker = Worker("EUR_USD", "M1", "192.168.0.115", 1546300800, "./models/")
    worker.run()
