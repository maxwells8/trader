import torch
import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
from environment import *
import networks
from networks import *
import pickle
import redis
import math
from zeus.zeus import Zeus

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, instrument, granularity, n_steps, models_loc):

        self.server = redis.Redis("localhost")
        self.zeus = Zeus(instrument, granularity)

        while True:
            # putting this while loop here because sometime the optimizer
            # is writing to the files and causes an exception
            self.market_encoder = AttentionMarketEncoder(device='cpu')
            self.decoder = Decoder()
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

    def add_bar(self, bar):
        time_state = torch.Tensor([bar.open, bar.high, bar.low, bar.close]).view(1, 1, -1)
        self.time_states.append(time_state)
        if len(self.time_states) > self.window:
            del self.time_states[0]
        spread = bar.spread

        if len(self.time_states) == self.window and self.steps_since_trade >= self.n_steps_future:


            time_states_ = torch.cat(self.time_states, dim=1).clone()
            mean = time_states_[:, :, :4].contiguous().view(1, -1).mean(1).view(1, 1, 1)
            std = time_states_[:, :, :4].contiguous().view(1, -1).std(1).view(1, 1, 1)
            time_states_[:, :, :4] = (time_states_[:, :, :4] - mean) / std
            spread_ = torch.Tensor([spread]).view(1, 1, 1) / std
            time_states_ = time_states_.transpose(0, 1)

            self.n_steps_future = np.random.randint(1, 120)

            market_encoding = self.market_encoder.forward(time_states_)
            advantage_ = self.decoder.forward(market_encoding, spread_, torch.Tensor([self.n_steps_future]).log())

            action = int(torch.argmax(advantage_.squeeze()))
            if action == 0:
                if self.pos != "Long":
                    if self.trade is not None:
                        self.zeus.close_trade(self.trade)
                    self.trade = self.zeus.place_trade(1000, "Long")
                self.pos = "Long"
            elif action == 1:
                if self.pos != "Short":
                    if self.trade is not None:
                        self.zeus.close_trade(self.trade)
                    self.trade = self.zeus.place_trade(1000, "Short")
                self.pos = "Short"
            else:
                if self.trade is not None:
                    self.zeus.close_trade(self.trade)
                self.trade = None
                self.pos = "Stay"
            self.steps_since_trade = 0

            print(advantage_)
            print(self.pos, action)
            print(self.zeus.unrealized_balance())
            print()

        else:
            self.steps_since_trade += 1


    def run(self):
        self.zeus.stream_bars(self.n_steps, self.add_bar)


if __name__ == "__main__":
    worker = Worker("NZD_USD", "M1", 7200*1, 'C:\\Users\\Preston\\Programming\\trader\\models\\')
    worker.run()
