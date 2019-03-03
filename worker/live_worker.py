import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
import networks
from networks import *
from environment import *
import redis
import msgpack
import math
from zeus.zeus import Zeus

assert False

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, instrument, granularity, models_loc):

        while True:
            self.market_encoder = CNNEncoder()
            self.actor_critic = ActorCritic()
            self.encoder_to_others = EncoderToOthers()
            try:
                self.market_encoder.load_state_dict(torch.load(models_loc + 'market_encoder.pt', map_location='cpu'))
                self.encoder_to_others.load_state_dict(torch.load(models_loc + 'encoder_to_others.pt', map_location='cpu'))
                self.actor_critic.load_state_dict(torch.load(models_loc + 'actor_critic.pt', map_location='cpu'))
                self.market_encoder = self.market_encoder.cpu()
                self.encoder_to_others = self.encoder_to_others.cpu()
                self.actor_critic = self.actor_critic.cpu()
                break
            except Exception:
                print("Failed to load models")
                time.sleep(0.1)

        self.models_loc = models_loc

        self.instrument = instrument
        self.granularity = granularity
        self.zeus = Zeus(instrument, granularity)
        self.live = False

        self.time_states = []

        self.window = networks.WINDOW

        self.tradeable_percentage = 1
        self.trade_percent = self.tradeable_percentage / 1000


    def add_bar(self, bar):
        time_state = [[[bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]]]

        if len(self.time_states) == 0 or time_state != self.time_states[-1]:
            self.time_states.append(time_state)
        else:
            return

        if len(self.time_states) == self.window + 1:
            del self.time_states[0]

        if len(self.time_states) == self.window and self.live:
            in_ = self.zeus.position_size()
            available_ = self.zeus.units_available()
            percent_in = (in_ / (abs(in_) + available_ + 1e-9)) / self.tradeable_percentage

            input_time_states = torch.Tensor(self.time_states).view(self.window, 1, networks.D_BAR).cpu()
            mean = input_time_states[:, 0, :4].mean()
            std = input_time_states[:, 0, :4].std()
            input_time_states[:, 0, :4] = (input_time_states[:, 0, :4] - mean) / std
            spread_normalized = bar.spread / std

            market_encoding = self.market_encoder.forward(input_time_states)
            market_encoding = self.encoder_to_others.forward(market_encoding, torch.Tensor([spread_normalized]), torch.Tensor([percent_in]))

            policy, value = self.actor_critic.forward(market_encoding)

            # action = torch.argmax(policy).item()
            action = torch.multinomial(policy, 1).item()

            def place_action(desired_percent):
                current_percent_in = percent_in * self.tradeable_percentage

                if desired_percent == 0 and current_percent_in != 0:
                    self.zeus.close_units(self.zeus.position_size())
                elif desired_percent > 0 and current_percent_in > 0:
                    if desired_percent > current_percent_in:
                        total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                        self.zeus.place_trade(int(abs(desired_percent - current_percent_in) * total_tradeable), "Long")
                    else:
                        total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                        self.zeus.close_units(int(abs((desired_percent - current_percent_in)) * total_tradeable))

                elif desired_percent > 0 and current_percent_in <= 0:
                    self.zeus.close_units(self.zeus.position_size())
                    total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                    self.zeus.place_trade(int(abs(desired_percent) * total_tradeable), "Long")

                elif desired_percent < 0 and current_percent_in > 0:
                    self.zeus.close_units(self.zeus.position_size())
                    total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                    self.zeus.place_trade(int(abs(desired_percent) * total_tradeable), "Short")

                elif desired_percent < 0 and current_percent_in <= 0:
                    if desired_percent <= current_percent_in:
                        total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                        self.zeus.place_trade(int(abs(desired_percent - current_percent_in) * total_tradeable), "Short")
                    else:
                        total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                        self.zeus.close_units(int(abs((desired_percent - current_percent_in)) * total_tradeable))

            change_amounts = {0:-100, 1:-50, 2:-10, 3:-5, 4:-1, 5:0, 6:1, 7:5, 8:10, 9:50, 10:100}
            if action in change_amounts:
                desired_percent_in = (percent_in * self.tradeable_percentage) + (self.trade_percent * change_amounts[action])
                desired_percent_in = np.clip(desired_percent_in, -self.tradeable_percentage, self.tradeable_percentage)
                place_action(desired_percent_in)

            print("instrument", self.instrument)
            print("purchased:", change_amounts[action])
            print("percent in:", round(percent_in, 5))
            for i, policy_ in enumerate(policy.tolist()[0]):
                print("probability {i}: {p}".format(i=i, p=round(policy_, 5)))
            print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

        torch.cuda.empty_cache()

    def run(self):
        self.market_encoder.eval()
        self.encoder_to_others.eval()
        self.actor_critic.eval()

        print("fetching last", self.window, "bars...")
        self.zeus.stream_bars(self.window, self.add_bar)

        print("going live...")
        self.live = True
        self.zeus = Zeus(self.instrument, self.granularity, live=True)
        self.zeus.stream_live(self.add_bar)
