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

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, name, instrument, granularity, models_loc, start, test=False):

        while True:
            self.market_encoder = LSTMCNNEncoder().cpu()
            self.encoder_to_others = EncoderToOthers().cpu()
            self.actor_critic = ActorCritic().cpu()
            self.encoder_to_others = EncoderToOthers().cpu()
            try:
                self.market_encoder.load_state_dict(torch.load(models_loc + 'market_encoder.pt'))
                self.encoder_to_others.load_state_dict(torch.load(models_loc + 'encoder_to_others.pt'))
                self.actor_critic.load_state_dict(torch.load(models_loc + 'actor_critic.pt'))
                self.market_encoder = self.market_encoder.cpu()
                self.encoder_to_others = self.encoder_to_others.cpu()
                self.actor_critic = self.actor_critic.cpu()
                break
            except Exception:
                print("Failed to load models")
                time.sleep(0.1)

        self.name = name
        self.models_loc = models_loc

        self.server = redis.Redis("localhost")
        self.instrument = instrument
        self.granularity = granularity
        self.zeus = Zeus(instrument, granularity)

        self.time_states = []
        self.percents_in = []
        self.spreads = []
        self.mus = []
        self.actions = []
        self.rewards = []

        self.total_actual_reward = 0
        self.reward_tau = float(self.server.get("reward_tau").decode("utf-8"))

        self.window = networks.WINDOW

        self.start = start
        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.p_new_proposal = float(self.server.get("p_new_proposal").decode("utf-8"))


        self.test = test
        if self.test:
            self.tradeable_percentage = 1
            self.steps_before_trajectory = 1440 - self.trajectory_steps
            self.trade_percent = self.tradeable_percentage / 1000

            self.actor_temp = 1
            self.proposer_temps = {"w":1, "mu":1, "sigma":1}
            self.proposer_gate_temp = 1
        else:
            self.tradeable_percentage = 0.01
            self.steps_before_trajectory = 240
            self.trade_percent = self.tradeable_percentage / 1000

            actor_base_temp = float(self.server.get("actor_temp").decode("utf-8"))
            self.actor_temp = np.random.exponential(actor_base_temp / np.log(2))


        self.n_steps_left = self.window + self.trajectory_steps + self.steps_before_trajectory
        self.i_step = 0
        self.steps_since_push = 0
        self.steps_between_experiences = 0

        self.prev_value = self.zeus.unrealized_balance()

    def add_bar(self, bar):
        time_state = [[[bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]]]

        if len(self.time_states) == 0 or time_state != self.time_states[-1]:
            self.time_states.append(time_state)
        else:
            return

        if len(self.time_states) >= self.window:
            in_ = self.zeus.position_size()
            available_ = self.zeus.units_available()
            percent_in = (in_ / (abs(in_) + available_ + 1e-9)) / self.tradeable_percentage

            input_time_states = torch.Tensor(self.time_states[-self.window:]).view(self.window, 1, networks.D_BAR).cpu()
            mean = input_time_states[:, 0, :4].mean()
            std = input_time_states[:, 0, :4].std()
            input_time_states[:, 0, :4] = (input_time_states[:, 0, :4] - mean) / std
            assert torch.isnan(input_time_states).sum() == 0
            spread_normalized = bar.spread / std

            market_encoding = self.market_encoder.forward(input_time_states)
            market_encoding = self.encoder_to_others.forward(market_encoding, torch.Tensor([spread_normalized]), torch.Tensor([percent_in]))

            policy, value = self.actor_critic.forward(market_encoding, temp=self.actor_temp)

            if self.test:
                action = torch.argmax(policy).item()
            else:
                action = torch.multinomial(policy, 1).item()

            mu = policy[0, action].item()

            def place_action(desired_percent):
                if desired_percent == 0 and percent_in != 0:
                    self.zeus.close_units(self.zeus.position_size())
                elif desired_percent > 0 and percent_in > 0:
                    if desired_percent > percent_in:
                        total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                        self.zeus.place_trade(int(abs(desired_percent - percent_in) * total_tradeable), "Long")
                    else:
                        total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                        self.zeus.close_units(int(abs((desired_percent - percent_in)) * total_tradeable))

                elif desired_percent > 0 and percent_in <= 0:
                    self.zeus.close_units(self.zeus.position_size())
                    total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                    self.zeus.place_trade(int(abs(desired_percent) * total_tradeable), "Long")

                elif desired_percent < 0 and percent_in > 0:
                    total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                    self.zeus.place_trade(int(abs(desired_percent) * total_tradeable), "Short")

                elif desired_percent < 0 and percent_in <= 0:
                    if desired_percent <= percent_in:
                        total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                        self.zeus.place_trade(int(abs(desired_percent - percent_in) * total_tradeable), "Short")
                    else:
                        total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                        self.zeus.close_units(int(abs((desired_percent - percent_in)) * total_tradeable))

            action_amounts = {0:1, 1:3, 2:5, 3:10, 4:-1, 5:-3, 6:-5, 7:-10}
            if action in action_amounts:
                desired_percent_in = np.clip(percent_in + self.trade_percent * action_amounts[action], -self.tradeable_percentage, self.tradeable_percentage)
                place_action(desired_percent_in)
            elif action == 8:
                place_action(0)

            new_val = self.zeus.unrealized_balance()
            self.total_actual_reward += new_val - self.prev_value
            reward = (new_val - self.prev_value) / (2000 * self.trade_percent)
            reward *= 10
            self.prev_value = new_val
            # if self.n_steps_left % 10 == action:
            #     reward = 1
            #     self.total_actual_reward += 1
            # else:
            #     reward = -1
            #     self.total_actual_reward += -1


            if self.test:
                reward_ema = self.server.get("test_reward_ema")
                reward_emsd = self.server.get("test_reward_emsd")
                if reward_ema != None:
                    reward_ema = float(reward_ema.decode("utf-8"))
                    reward_emsd = float(reward_emsd.decode("utf-8"))
                else:
                    reward_ema = 0
                    reward_emsd = 0
                in_ = self.zeus.position_size()
                available_ = self.zeus.units_available()
                percent_in_ = (in_ / (abs(in_) + available_ + 1e-9)) / self.tradeable_percentage

                print("step: {s} \
                \npercent in: {p_in} \
                \naction: {a} \
                \nunrealized_balance: {u_b} \
                \npolicy: {p} \
                \nvalue: {v} \
                \nrewards: {r} \
                \nreward_ema: {ema} \
                \nreward_emsd: {emsd}\n".format(s=self.i_step,
                                        p_in=round(percent_in_, 8),
                                        a=action,
                                        u_b=round(new_val, 2),
                                        p=[round(policy_, 5) for policy_ in policy[0].tolist()],
                                        v=round(value.item(), 2),
                                        r=round(self.total_actual_reward, 2),
                                        ema=round(reward_ema, 2),
                                        emsd=round(reward_emsd, 2)))


            if not self.test:
                reward_ema = self.server.get("reward_ema").decode("utf-8")
                if reward_ema == 'None':
                    self.server.set("reward_ema", reward)
                    self.server.set("reward_emsd", 0)
                else:
                    reward_ema = float(reward_ema)
                    reward_emsd = float(self.server.get("reward_emsd").decode("utf-8"))
                    delta = reward - reward_ema
                    self.server.set("reward_ema", reward_ema + self.reward_tau * delta)
                    self.server.set("reward_emsd", math.sqrt((1 - self.reward_tau) * (reward_emsd**2 + self.reward_tau * (delta**2))))

            self.percents_in.append(percent_in)
            self.spreads.append(bar.spread)
            self.rewards.append(reward)
            self.actions.append(action)
            self.mus.append(mu)

            if len(self.time_states) == self.window + self.trajectory_steps + 1:
                del self.time_states[0]
            if len(self.percents_in) == self.trajectory_steps + 1:
                del self.percents_in[0]
                del self.spreads[0]
                del self.rewards[0]
                del self.actions[0]
                del self.mus[0]

            if (self.steps_since_push >= self.steps_between_experiences) and not self.test and len(self.time_states) == self.window + self.trajectory_steps:

                experience = Experience(
                time_states=self.time_states,
                percents_in=self.percents_in,
                spreads=self.spreads,
                mus=self.mus,
                place_actions=self.actions,
                rewards=self.rewards
                )

                experience = msgpack.packb(experience, use_bin_type=True)
                n_experiences = self.server.llen("experience")
                if n_experiences > 0:
                    try:
                        loc = np.random.randint(0, n_experiences)
                        ref = self.server.lindex("experience", loc)
                        self.server.linsert("experience", "before", ref, experience)
                    except redis.exceptions.DataError:
                        self.server.lpush("experience", experience)
                else:
                    self.server.lpush("experience", experience)
                self.steps_since_push = 0
            else:
                self.steps_since_push += 1

            self.i_step += 1

        self.n_steps_left -= 1
        torch.cuda.empty_cache()

    def run(self):
        self.market_encoder.eval()
        self.encoder_to_others.eval()
        self.actor_critic.eval()

        t0 = time.time()
        while self.n_steps_left > 0:
            n_seconds = min(self.n_steps_left, 500) * 60
            if self.granularity == "M5":
                n_seconds *= 5
            self.zeus.stream_range(self.start, self.start + n_seconds, self.add_bar)
            self.start += n_seconds

        print(("time: {time}, "
                "total rewards (percent): {reward}, "
                "actor temp: {actor_temp}, "
                "steps: {steps} ").format(
                    time=round(time.time()-t0, 2),
                    reward=round(100 * self.total_actual_reward / (2000 * self.tradeable_percentage), 5),
                    actor_temp=round(self.actor_temp, 3),
                    steps=self.steps_before_trajectory
                )
        )

        if self.test:
            reward_tau = float(self.server.get("test_reward_tau").decode("utf-8"))
            reward_ema = self.server.get("test_reward_ema")
            if reward_ema == None:
                self.server.set("test_reward_ema", self.total_actual_reward)
                self.server.set("test_reward_emsd", 0)
            else:
                reward_ema = float(reward_ema.decode("utf-8"))
                reward_emsd = float(self.server.get("test_reward_emsd").decode("utf-8"))
                delta = self.total_actual_reward - reward_ema
                self.server.set("test_reward_ema", reward_ema + reward_tau * delta)
                self.server.set("test_reward_emsd", math.sqrt((1 - reward_tau) * (reward_emsd**2 + reward_tau * (delta**2))))

        return self.total_actual_reward

Experience = namedtuple('Experience', ('time_states',
                                       'percents_in',
                                       'spreads',
                                       'mus',
                                       'place_actions',
                                       'rewards'))
