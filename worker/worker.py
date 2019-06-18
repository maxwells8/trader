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
import pickle
from zeus.zeus import Zeus

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, name, instrument, granularity, server_host, start, test=False):

        if not test:
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        self.name = name
        self.instrument = instrument
        self.granularity = granularity
        self.server = redis.Redis(server_host)

        while True:
            if not test:
                self.market_encoder = Encoder().cpu()
                self.actor_critic = ActorCritic().cpu()

                try:
                    MEN_compressed_state_dict = self.server.get("market_encoder")
                    ACN_compressed_state_dict = self.server.get("actor_critic")

                    MEN_state_dict_buffer = pickle.loads(MEN_compressed_state_dict)
                    ACN_state_dict_buffer = pickle.loads(ACN_compressed_state_dict)

                    MEN_state_dict_buffer.seek(0)
                    ACN_state_dict_buffer.seek(0)

                    self.market_encoder.load_state_dict(torch.load(MEN_state_dict_buffer, map_location='cpu'))
                    self.actor_critic.load_state_dict(torch.load(ACN_state_dict_buffer, map_location='cpu'))

                    break
                except Exception as e:
                    print("Failed to load models")
                    time.sleep(0.1)

            else:
                self.market_encoder = Encoder().cuda()
                self.actor_critic = ActorCritic().cuda()
                try:
                    MEN_compressed_state_dict = self.server.get("market_encoder")
                    ACN_compressed_state_dict = self.server.get("actor_critic")

                    MEN_state_dict_buffer = pickle.loads(MEN_compressed_state_dict)
                    ACN_state_dict_buffer = pickle.loads(ACN_compressed_state_dict)

                    MEN_state_dict_buffer.seek(0)
                    ACN_state_dict_buffer.seek(0)

                    self.market_encoder.load_state_dict(torch.load(MEN_state_dict_buffer, map_location='cuda'))
                    self.actor_critic.load_state_dict(torch.load(ACN_state_dict_buffer, map_location='cuda'))
                    break
                except Exception:
                    print("Failed to load models")
                    time.sleep(0.1)

        self.market_encoder.eval()
        self.actor_critic.eval()

        if test:
            n = 0
            for network in [self.market_encoder, self.actor_critic]:
                for param in network.parameters():
                    n += np.prod(param.size())
            print("number of parameters:", n)

        self.time_states = []
        self.all_time_states = []
        self.percents_in = []
        self.spreads = []
        self.mus = []
        self.actions = []
        self.rewards = []

        self.total_actual_reward = 0
        self.total_corrected_reward = 0

        self.window = networks.WINDOW

        self.start = start
        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.i_step = 0
        self.steps_since_push = 0
        self.n_experiences = 0
        self.steps_between_experiences = 10

        self.test = test
        if self.test:
            self.zeus = Zeus(instrument, granularity, margin=1)
            self.tradeable_percentage = 1
            self.n_steps_left = self.window + (1440 * 5)
            self.n_total_experiences = 0

            self.actor_temp = 1
        else:
            self.zeus = Zeus(instrument, granularity, margin=1)
            self.tradeable_percentage = 1
            self.n_total_experiences = 100
            self.n_steps_left = self.window + (self.steps_between_experiences + self.trajectory_steps) * self.n_total_experiences

            actor_base_temp = float(self.server.get("actor_temp").decode("utf-8"))
            self.actor_temp = np.random.exponential(actor_base_temp / np.log(2))

        self.trade_percent = self.tradeable_percentage / 100
        self.trade_units = 0
        self.prev_value = self.zeus.unrealized_balance()

        self.plot = self.test and False
        if self.plot:
            self.all_times = []
            self.long_times = []
            self.short_times = []
            self.long_bars = []
            self.short_bars = []
            self.all_bars = []

    def add_bar(self, bar):
        time_state = [[[bar.open, bar.high, bar.low, bar.close]]]

        if (bar.date > 1546300800) or (not self.test and self.n_experiences == self.n_total_experiences):
            self.quit()

        if self.i_step == 0:
            self.trade_units = self.zeus.units_available()

        if len(self.time_states) == 0 or time_state != self.time_states[-1]:
            self.time_states.append(time_state)
            self.all_time_states.append([bar.open, bar.high, bar.low, bar.close])
        else:
            return

        if len(self.time_states) >= self.window:
            if self.plot:
                self.all_times.append(self.i_step)
                self.all_bars.append(bar.close)

            in_ = self.zeus.position_size()
            available_ = self.zeus.units_available()
            percent_in = (in_ / (abs(in_) + available_ + 1e-9)) / self.tradeable_percentage

            input_time_states = torch.Tensor(self.time_states[-self.window:]).view(1, self.window, networks.D_BAR)
            assert torch.isnan(input_time_states).sum() == 0

            market_encoding = self.market_encoder(input_time_states, torch.Tensor([bar.spread]), torch.Tensor([percent_in]))
            policy, value = self.actor_critic(market_encoding, temp=self.actor_temp)

            action = torch.multinomial(policy, 1).item()

            if self.plot:
                if action == 0:
                    self.short_times.append(self.i_step)
                    self.short_bars.append(bar.close)
                elif action == 2:
                    self.long_times.append(self.i_step)
                    self.long_bars.append(bar.close)

            mu = policy[0, action].item()

            def place_action(desired_percent):
                current_percent_in = percent_in * self.tradeable_percentage

                if desired_percent == 0 and current_percent_in != 0:
                    self.zeus.close_units(self.zeus.position_size())
                elif desired_percent > 0 and current_percent_in > 0:
                    total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                    if desired_percent > current_percent_in:
                        self.zeus.place_trade(int(abs(desired_percent - current_percent_in) * total_tradeable), "Long")
                    else:
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
                    total_tradeable = abs(self.zeus.position_size()) + self.zeus.units_available()
                    if desired_percent <= current_percent_in:
                        self.zeus.place_trade(int(abs(desired_percent - current_percent_in) * total_tradeable), "Short")
                    else:
                        self.zeus.close_units(int(abs((desired_percent - current_percent_in)) * total_tradeable))

            val_before_action = self.zeus.unrealized_balance()
            change_amounts = {0:-200, 2:200}
            if action in change_amounts:
                desired_percent_in = (percent_in * self.tradeable_percentage) + (self.trade_percent * change_amounts[action])
                # desired_percent_in = (percent_in * self.tradeable_percentage) + (self.trade_percent * change_amounts[action] * self.trade_units / (abs(in_) + available_))
                desired_percent_in = np.clip(desired_percent_in, -self.tradeable_percentage, self.tradeable_percentage)
                place_action(desired_percent_in)

            if action == 1:
                # if abs(percent_in) > 0.05:
                #     desired_percent_in = (percent_in * self.tradeable_percentage)
                #     if percent_in > 0:
                #         desired_percent_in -= self.trade_percent * 5
                #     elif percent_in < 0:
                #         desired_percent_in += self.trade_percent * 5
                # else:
                #     desired_percent_in = 0
                desired_percent_in = 0

                place_action(desired_percent_in)

            new_val = self.zeus.unrealized_balance()
            self.total_actual_reward += new_val - self.prev_value
            self.total_corrected_reward += val_before_action - self.prev_value
            # reward = (new_val - self.prev_value) / (2000 * self.trade_percent)
            reward = (val_before_action - self.prev_value) / (2000 * self.trade_percent)
            reward *= 100
            self.prev_value = new_val


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

                print('-----------------------------------------------------------------------------')
                print("step: {s} \
                \n\t\t\t\t\t\tpercent in: {p_in} \
                \naction: {a} \
                \nunrealized_balance: {u_b} \
                \npolicy: {p} \
                \nvalue: {v} \
                \nactual rewards: {a_r} \
                \ncorrected rewards: {c_r} \
                \nreward_ema: {ema} \
                \nreward_emsd: {emsd} \
                \nbar close: {close} \
                \nwindow std: {std} \
                \ninstrument: {ins} \
                \nstart: {start} \n".format(s=self.i_step,
                                        p_in=round(percent_in_, 8),
                                        a=action,
                                        u_b=round(new_val, 5),
                                        p=[round(policy_, 5) for policy_ in policy[0].tolist()],
                                        v=round(value.item(), 5),
                                        a_r=round(self.total_actual_reward, 5),
                                        c_r=round(self.total_corrected_reward, 5),
                                        ema=round(reward_ema, 5),
                                        emsd=round(reward_emsd, 5),
                                        std=round(input_time_states[:, 0, :4].std().item(), 5),
                                        close=bar.close,
                                        ins=self.instrument,
                                        start=self.start))

                print('-----------------------------------------------------------------------------')
                print('*****************************************************************************')
                if self.i_step % 1000 == 0:
                    try:
                        MEN_compressed_state_dict = self.server.get("market_encoder")
                        ACN_compressed_state_dict = self.server.get("actor_critic")

                        MEN_state_dict_buffer = pickle.loads(MEN_compressed_state_dict)
                        ACN_state_dict_buffer = pickle.loads(ACN_compressed_state_dict)

                        MEN_state_dict_buffer.seek(0)
                        ACN_state_dict_buffer.seek(0)

                        self.market_encoder.load_state_dict(torch.load(MEN_state_dict_buffer, map_location='cuda'))
                        self.actor_critic.load_state_dict(torch.load(ACN_state_dict_buffer, map_location='cuda'))
                    except Exception as e:
                        print("Failed to load models")

            # print("time_state[0]", self.time_states[-self.window])
            # print("percent_in", percent_in)
            # print("spread", bar.spread)
            # print("place_action", action)
            # print("policy", policy)
            # print("reward", reward)
            # print()

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

            if (self.steps_since_push == self.steps_between_experiences) and (not self.test) and len(self.time_states) == self.window + self.trajectory_steps:
                experience = Experience(
                time_states=self.time_states,
                percents_in=self.percents_in,
                spreads=self.spreads,
                mus=self.mus,
                place_actions=self.actions,
                rewards=self.rewards
                )

                experience = msgpack.packb(experience, use_bin_type=True)
                # self.add_to_replay_buffer(experience)
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
                self.n_experiences += 1
                self.steps_since_push = 1

                p_rand_percent_in = 0
                if np.random.rand() < p_rand_percent_in:
                    # desired_percent_in = np.random.normal(0, 0.1)
                    desired_percent_in = np.random.rand() * 2 - 1
                    desired_percent_in *= self.tradeable_percentage
                    desired_percent_in = np.clip(desired_percent_in, -self.tradeable_percentage, self.tradeable_percentage)
                    place_action(desired_percent_in)
                self.prev_value = self.zeus.unrealized_balance()
            else:
                self.steps_since_push += 1

            self.i_step += 1

        self.n_steps_left -= 1
        # torch.cuda.empty_cache()
        self.t_final_prev = time.time()

    def run(self):
        self.t0 = time.time()

        n = 0
        start = self.start
        while (self.test and self.n_steps_left > 0) or (not self.test and self.n_experiences < self.n_total_experiences):
            if not self.test:
                n_seconds = (self.n_total_experiences - self.n_experiences) * (self.steps_between_experiences + self.trajectory_steps) * 60
            else:
                n_seconds = self.n_steps_left * 60
            if self.granularity == "M5":
                n_seconds *= 5
            if self.test:
                print("starting new stream")
            self.zeus.stream_range(start, start + n_seconds, self.add_bar)
            start += n_seconds
            n += 1

        self.quit()

    def quit(self):
        print(("time: {time}, "
                "rewards: {reward} %, "
                "temp: {actor_temp}, "
                "n exp: {n_experiences}, "
                "instr: {instrument}, "
                "start: {start}").format(
                    time=round(time.time()-self.t0, 2),
                    reward=round(100 * self.total_corrected_reward / (2000 * self.tradeable_percentage), 5),
                    actor_temp=round(self.actor_temp, 3),
                    n_experiences=self.n_experiences,
                    instrument=self.instrument,
                    start=self.start
                )
        )

        if self.test:

            if self.plot:
                import matplotlib.pyplot as plt
                plt.plot(self.all_times, self.all_bars)
                plt.scatter(np.array(self.long_times), self.long_bars, c='g', alpha=1)
                plt.scatter(np.array(self.short_times), self.short_bars, c='r', alpha=1)
                plt.show()

            reward_tau = float(self.server.get("test_reward_tau").decode("utf-8"))
            reward_ema = self.server.get("test_reward_ema")
            if reward_ema == None:
                self.server.set("test_reward_ema", self.total_corrected_reward)
                self.server.set("test_reward_emsd", 0)
            else:
                reward_ema = float(reward_ema.decode("utf-8"))
                reward_emsd = float(self.server.get("test_reward_emsd").decode("utf-8"))
                delta = self.total_corrected_reward - reward_ema
                self.server.set("test_reward_ema", reward_ema + reward_tau * delta)
                self.server.set("test_reward_emsd", math.sqrt((1 - reward_tau) * (reward_emsd**2 + reward_tau * (delta**2))))

        quit()

    def add_to_replay_buffer(self, compressed_experience):
        # try:
        #     loc = np.random.randint(0, replay_size) if replay_size > 0 else 0
        #     ref = self.server.lindex("replay_buffer", loc)
        #     self.server.linsert("replay_buffer", "before", ref, compressed_experience)
        # except redis.exceptions.DataError:
        self.server.lpush("replay_buffer", compressed_experience)

        max_replay_size = int(self.server.get("replay_buffer_size").decode("utf-8"))
        replay_size = self.server.llen("replay_buffer")
        while replay_size - 1 > max_replay_size:
            replay_size = self.server.llen("replay_buffer")
            try:
                loc = replay_size - 1
                ref = self.server.lindex("replay_buffer", loc)
                self.server.lrem("replay_buffer", -1, ref)
            except Exception as e:
                pass

Experience = namedtuple('Experience', ('time_states',
                                       'percents_in',
                                       'spreads',
                                       'mus',
                                       'place_actions',
                                       'rewards'))
