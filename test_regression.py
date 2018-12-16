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
            self.decoder = Decoder().cpu()
            try:
                self.market_encoder.load_state_dict(torch.load(models_loc + 'market_encoder.pt'))
                self.decoder.load_state_dict(torch.load(models_loc + 'decoder.pt'))
                self.market_encoder = self.market_encoder.cpu()
                self.decoder = self.decoder.cpu()
                break
            except Exception as e:
                print(e)

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

    def add_bar(self, bar):
        time_state = torch.Tensor([bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]).view(1, 1, -1)
        self.time_states.append(time_state)
        if len(self.time_states) > self.window:
            del self.time_states[0]
        spread = bar.spread
        if len(self.time_states) == self.window and self.steps_since_trade >= self.n_steps_future:

            if self.trade is not None:
                self.zeus.close_trade(self.trade)

            time_states_ = torch.cat(self.time_states, dim=1).clone()
            mean = time_states_[:, :, :4].contiguous().view(1, -1).mean(1).view(1, 1, 1)
            std = time_states_[:, :, :4].contiguous().view(1, -1).std(1).view(1, 1, 1)
            time_states_[:, :, :4] = (time_states_[:, :, :4] - mean) / std
            spread_ = torch.Tensor([spread]).view(1, 1, 1) / std
            time_states_ = time_states_.transpose(0, 1)

            self.n_steps_future = np.random.randint(1, 60)
            self.n_steps_future = 30
            print(self.n_steps_future)

            market_encoding = self.market_encoder.forward(time_states_)
            value_ = self.decoder.forward(market_encoding, spread_, torch.Tensor([self.n_steps_future]).log())
            print(value_)

            # action = int(torch.argmax(torch.cat([value_.squeeze(), torch.Tensor([0])], dim=0)))
            # print("calculated with 0")
            action = int(torch.argmax(value_.squeeze()))
            print("calculated without 0")
            # action = np.random.randint(0, 2)
            # print("random")

            if action == 0:
                self.n_buy += 1
                if self.pos != "Long":
                    self.trade = self.zeus.place_trade(1000, "Long")
                self.pos = "Long"
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            elif action == 1:
                self.n_sell += 1
                if self.pos != "Short":
                    self.trade = self.zeus.place_trade(1000, "Short")
                self.pos = "Short"
                print("-------------------------------------------------------")
            else:
                self.n_stay += 1
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
    worker = Worker("NZD_USD", "M1", 7200*52, 'models\\')
    worker.run()
