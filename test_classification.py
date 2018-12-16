import torch
import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
from environment import *
import networks
from networks import *
import redis
import math
from zeus.zeus import Zeus

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)
np.random.seed(0)

class Worker(object):

    def __init__(self, instrument, granularity, n_steps, models_loc):

        self.server = redis.Redis("localhost")
        self.instrument = instrument
        self.granularity = granularity
        self.zeus = Zeus(instrument, granularity)

        while True:
            # putting this while loop here because sometime the optimizer
            # is writing to the files and causes an exception
            # self.market_encoder = AttentionMarketEncoder(device='cpu')
            self.market_encoder = CNNEncoder().cpu()
            self.decoder = ClassifierDecoder().cpu()
            try:
                self.market_encoder.load_state_dict(torch.load(models_loc + 'market_encoder.pt'))
                self.decoder.load_state_dict(torch.load(models_loc + 'decoder.pt'))
                self.market_encoder = self.market_encoder.cpu()
                self.decoder = self.decoder.cpu()
                break
            except Exception as e:
                print(e)

        self.trajectory_steps = float(self.server.get("trajectory_steps").decode("utf-8"))
        self.window = networks.WINDOW
        self.n_steps = n_steps

        self.time_states = []

        self.trade = None
        self.pos = None

        self.n_steps_future = 1
        self.steps_since_trade = 1

        self.cur_value = self.zeus.unrealized_balance()
        self.prev_value = self.cur_value

        self.n = 0
        self.n_profit = 0
        self.n_loss = 0
        self.n_buy = 0
        self.n_sell = 0
        self.n_stay = 0

        self.profit = 0

        self.stop_loss = None
        self.take_profit = None

    def add_bar(self, bar):
        if self.pos == "Long" and self.trade is not None:
            if bar.close > self.take_profit or bar.close < self.stop_loss:
                self.zeus.close_trade(self.trade)
                self.trade = None
        elif self.pos == "Short" and self.trade is not None:
            if bar.close < self.take_profit or bar.close > self.stop_loss:
                self.zeus.close_trade(self.trade)
                self.trade = None

        time_state = torch.Tensor([bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]).view(1, 1, -1)
        self.time_states.append(time_state)
        if len(self.time_states) > self.window:
            del self.time_states[0]
        spread = bar.spread
        if len(self.time_states) == self.window and (self.steps_since_trade >= self.n_steps_future or (self.pos != "Stay" and self.trade is None)):
            print(bar)
            print(self.trade)

            time_states_ = torch.cat(self.time_states, dim=1).clone()
            mean = time_states_[:, :, :4].contiguous().view(1, -1).mean(1).view(1, 1, 1)
            std = time_states_[:, :, :4].contiguous().view(1, -1).std(1).view(1, 1, 1)
            time_states_[:, :, :4] = (time_states_[:, :, :4] - mean) / std
            spread_ = torch.Tensor([spread]).view(1, 1, 1) / std
            time_states_ = time_states_.transpose(0, 1)

            self.n_steps_future = self.trajectory_steps
            # confidence_interval = np.random.exponential(0.0015)
            confidence_interval = 0.001
            print("confidence:", confidence_interval)

            market_encoding = self.market_encoder.forward(time_states_)
            probabilities = self.decoder.forward(market_encoding, spread_ * std, std, torch.Tensor([confidence_interval]))
            print(probabilities)

            # action = int(torch.argmax(probabilities.squeeze()))
            # print("argmax")
            # action = int(torch.argmax(probabilities.squeeze()[:2]))
            # print("argmax without hold")
            action = int(torch.multinomial(probabilities.squeeze(), 1))
            print("sampled")
            # action = int(torch.multinomial(probabilities.squeeze()[:2], 1))
            # print("sampled without hold")
            # action = np.random.randint(0, 2)
            # print("random")

            if action == 0:
                self.stop_loss = self.time_states[-1][0, 0, 3] * (1 - confidence_interval) - spread
                self.take_profit = self.time_states[-1][0, 0, 3] * (1 + confidence_interval) - spread
                self.n_buy += 1
                if self.pos != "Long":
                    if self.trade is not None:
                        self.zeus.close_trade(self.trade)
                    self.trade = self.zeus.place_trade(100, "Long")
                self.pos = "Long"
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            elif action == 1:
                self.stop_loss = self.time_states[-1][0, 0, 3] * (1 + confidence_interval) - spread
                self.take_profit = self.time_states[-1][0, 0, 3] * (1 - confidence_interval) - spread
                self.n_sell += 1
                if self.pos != "Short":
                    if self.trade is not None:
                        self.zeus.close_trade(self.trade)
                    self.trade = self.zeus.place_trade(100, "Short")
                self.pos = "Short"
                print("-------------------------------------------------------")
            else:
                self.stop_loss = None
                self.take_profit = None
                self.n_stay += 1
                if self.trade is not None:
                    self.zeus.close_trade(self.trade)
                    self.trade = None
                self.pos = "Stay"
                print("///////////////////////////////////////////////////////")


            self.cur_value = self.zeus.unrealized_balance()
            print(self.cur_value)

            self.steps_since_trade = 0
            if self.prev_value < self.cur_value:
                self.n_profit += 1
            elif self.prev_value > self.cur_value:
                self.n_loss += 1
            self.profit += self.cur_value - self.prev_value
            self.n += 1

            print("profit this trade:", self.cur_value - self.prev_value)
            print("profit:", self.profit)
            print("profit per trade:", self.profit / self.n)
            print("gain per trade:", self.profit / (self.n * 1000))
            print("p loss:", self.n_loss / self.n)
            print("p profit:", self.n_profit / self.n)
            print("total trades:", self.n)
            print("p buy:", self.n_buy / self.n)
            print("p sell:", self.n_sell / self.n)
            print("p stay:", self.n_stay / self.n)
            print()

            self.prev_value = self.cur_value

        else:
            self.steps_since_trade += 1

    def run(self):
        self.zeus.stream_bars(self.n_steps, self.add_bar)


if __name__ == "__main__":
    worker = Worker("GBP_USD", "M1", 7200*12, 'models\\')
    worker.run()
