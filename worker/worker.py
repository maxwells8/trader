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
        self.exp_time_states = []
        self.exp_percents_in = []
        self.exp_trades_open = []
        self.exp_spreads = []
        self.exp_mus = []
        self.exp_actions = []
        self.exp_rewards = []

        self.action_last_step = False
        self.last_bar_date = -1

        self.total_rewards = 0

        self.window = networks.WINDOW

        self.start = start
        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.i_step = 0
        self.steps_since_push = 0
        self.n_experiences = 0
        self.steps_between_experiences = self.trajectory_steps

        self.test = test
        if self.test:
            self.zeus = Zeus(instrument, granularity, margin=1)
            self.tradeable_percentage = 1
            self.n_steps_left = self.window + (1440 * 5)
            self.n_total_experiences = 0

            self.actor_temp = 1

            self.trade_steps = []
        else:
            self.zeus = Zeus(instrument, granularity, margin=1)
            self.tradeable_percentage = 1
            self.n_total_experiences = 100
            self.n_steps_left = self.window + (self.steps_between_experiences + self.trajectory_steps) * self.n_total_experiences

            actor_base_temp = float(self.server.get("actor_temp").decode("utf-8"))
            self.actor_temp = np.random.exponential(actor_base_temp / np.log(2))

        self.trade_percent = self.tradeable_percentage / 100
        self.trade_units = 0

        self.long_trades = []
        self.short_trades = []

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

        if (self.test and self.n_steps_left == 0) or (not self.test and bar.date > 1546300800) or (not self.test and self.n_experiences == self.n_total_experiences):
            self.quit()

        if self.i_step == 0:
            self.trade_units = self.zeus.units_available()

        if len(self.time_states) == 0 or bar.date != self.last_bar_date:
            self.time_states.append(time_state)
            self.last_bar_date = bar.date
        else:
            return

        if len(self.time_states) == self.window + 1:
            del self.time_states[0]

        if len(self.time_states) == self.window:
            if self.plot:
                self.all_times.append(self.i_step)
                self.all_bars.append(bar.close)

            reward = 0

            test_with_spread = True
            workers_with_spread = False

            # process short trades
            completed_trades = set()
            for i, trade in enumerate(self.short_trades):
                trade_open = trade['open']
                if (self.test and test_with_spread) or (not self.test and workers_with_spread):
                    trade_open -= trade['spread']

                if trade_open - bar.low > trade['tp']:
                    reward += trade['tp']
                    completed_trades.add(i)
                if bar.high - trade_open > trade['sl']:
                    reward -= trade['sl']
                    completed_trades.add(i)

            completed_trades = sorted(list(completed_trades))
            completed_trades.reverse()
            for i in completed_trades:
                if self.test:
                    self.trade_steps.append(self.i_step - self.short_trades[i]['step'])
                del self.short_trades[i]

            # process long trades
            completed_trades = set()
            for i, trade in enumerate(self.long_trades):
                trade_open = trade['open']
                if (self.test and test_with_spread) or (not self.test and workers_with_spread):
                    trade_open += trade['spread']

                if bar.high - trade_open > trade['tp']:
                    reward += trade['tp']
                    completed_trades.add(i)
                if trade_open - bar.low > trade['sl']:
                    reward -= trade['sl']
                    completed_trades.add(i)

            completed_trades = sorted(list(completed_trades))
            completed_trades.reverse()
            for i in completed_trades:
                if self.test:
                    self.trade_steps.append(self.i_step - self.long_trades[i]['step'])
                del self.long_trades[i]

            reward *= 1000
            self.total_rewards += reward

            percent_in = -1 * len(self.short_trades) + 1 * len(self.long_trades)

            if percent_in == 0:

                if self.action_last_step:
                    # print("<reward>")
                    # print(self.i_step, reward)
                    # print("</reward>")

                    self.exp_rewards.append(reward)
                    self.action_last_step = False

                    if len(self.exp_rewards) == self.trajectory_steps:
                        del self.exp_rewards[0]

                trade_open = bar.close
                if len(self.short_trades) > 0:
                    trade_open = self.short_trades[0]["open"]
                elif len(self.long_trades) > 0:
                    trade_open = self.long_trades[0]["open"]

                input_time_states = torch.Tensor(self.time_states[-self.window:]).view(1, self.window, networks.D_BAR)
                assert torch.isnan(input_time_states).sum() == 0

                market_encoding = self.market_encoder(input_time_states, torch.Tensor([bar.spread]), torch.Tensor([percent_in]), torch.Tensor([trade_open]))
                policy, value = self.actor_critic(market_encoding, temp=self.actor_temp)

                if self.test and False:
                    action = np.random.randint(0, 3)
                    # action = torch.multinomial(policy, 1).item()
                else:
                    action = torch.multinomial(policy, 1).item()

                # print('<input>')
                # print(self.i_step, action, torch.Tensor(input_time_states.tolist()))
                # print('</input>')

                if self.plot:
                    if action == 0:
                        self.short_times.append(self.i_step)
                        self.short_bars.append(bar.close)
                    elif action == 1:
                        self.long_times.append(self.i_step)
                        self.long_bars.append(bar.close)

                mu = policy[0, action].item()

                self.action_last_step = True

                self.exp_time_states.append(input_time_states.tolist())
                self.exp_percents_in.append(percent_in)
                self.exp_trades_open.append(trade_open)
                self.exp_spreads.append(bar.spread)

                if len(self.exp_time_states) == self.trajectory_steps + 1:
                    del self.exp_time_states[0]
                    del self.exp_percents_in[0]
                    del self.exp_trades_open[0]
                    del self.exp_spreads[0]

                # process action
                open = bar.close
                spread = bar.spread
                step = self.i_step
                if self.test:
                    tp = 5 / 10000
                    sl = 5 / 10000
                else:
                    tp = 5 / 10000
                    sl = 5 / 10000
                new_trade = {'open':open, 'spread':spread, 'step':step, 'tp':tp, 'sl':sl}

                max_trades = 1

                if action == 0:
                    # if len(self.long_trades) > 0:
                    #     trade = self.long_trades[0]
                    #     trade_open = trade['open']
                    #     if (self.test and test_with_spread) or (not self.test and workers_with_spread):
                    #         trade_open += trade['spread']
                    #
                    #     reward += bar.close - trade_open
                    #     del self.long_trades[0]
                    # elif len(self.short_trades) < max_trades:
                    #     self.short_trades.append(new_trade)
                    if len(self.short_trades) < max_trades and len(self.long_trades) == 0:
                        self.short_trades.append(new_trade)

                elif action == 1:
                    # if len(self.short_trades) > 0:
                    #     trade = self.short_trades[0]
                    #     trade_open = trade['open']
                    #     if (self.test and test_with_spread) or (not self.test and workers_with_spread):
                    #         trade_open -= trade['spread']
                    #
                    #     reward += trade_open - bar.close
                    #     del self.short_trades[0]
                    # elif len(self.long_trades) < max_trades:
                    #     self.long_trades.append(new_trade)
                    if len(self.long_trades) < max_trades and len(self.short_trades) == 0:
                        self.long_trades.append(new_trade)

                if self.test:
                    reward_ema = self.server.get("test_reward_ema")
                    reward_emsd = self.server.get("test_reward_emsd")
                    if reward_ema != None:
                        reward_ema = float(reward_ema.decode("utf-8"))
                        reward_emsd = float(reward_emsd.decode("utf-8"))
                    else:
                        reward_ema = 0
                        reward_emsd = 0

                    print('-----------------------------------------------------------------------------')
                    print("step: {s} \
                    \naction: {a} \
                    \ndirection: {dir} \
                    \npolicy: {p} \
                    \nvalue: {v} \
                    \ntotal rewards: {t_r} \
                    \nreward_ema: {ema} \
                    \nreward_emsd: {emsd} \
                    \nbar close: {close} \
                    \nwindow std: {std} \
                    \ninstrument: {ins} \
                    \nstart: {start} \n".format(s=self.i_step,
                                            a=action,
                                            dir=percent_in,
                                            p=[round(policy_, 5) for policy_ in policy[0].tolist()],
                                            v=round(value.item(), 5),
                                            t_r=self.total_rewards,
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

                if (self.steps_since_push == self.steps_between_experiences) and (not self.test) and len(self.exp_time_states) == self.trajectory_steps:
                    experience = Experience(
                    time_states=self.exp_time_states,
                    percents_in=self.exp_percents_in,
                    trades_open=self.exp_trades_open,
                    spreads=self.exp_spreads,
                    mus=self.exp_mus,
                    place_actions=self.exp_actions,
                    rewards=self.exp_rewards
                    )

                    # print('<experience>')
                    # print(self.i_step, self.exp_actions[0], self.exp_rewards[0], torch.Tensor(self.exp_time_states[0]), torch.Tensor(self.exp_time_states[1]))
                    # print('</experience>')

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

                    self.exp_time_states = []
                    self.exp_percents_in = []
                    self.exp_trades_open = []
                    self.exp_spreads = []
                    self.exp_mus = []
                    self.exp_actions = []
                    self.exp_rewards = []
                    self.action_last_step = False

                else:
                    self.steps_since_push += 1
                    self.action_last_step = True

                    self.exp_actions.append(action)
                    self.exp_mus.append(mu)
                    if len(self.exp_actions) == self.trajectory_steps:
                        del self.exp_actions[0]
                        del self.exp_mus[0]

            self.i_step += 1

        self.n_steps_left -= 1
        # torch.cuda.empty_cache()
        self.t_final_prev = time.time()

    def run(self):
        self.t0 = time.time()

        n = 0
        start = self.start
        while (self.test and self.n_steps_left > 0) or (not self.test and self.n_experiences < self.n_total_experiences):
            if self.test:
                n_seconds = min(self.n_steps_left, 7200) * 60
            else:
                n_seconds = (self.n_total_experiences - self.n_experiences) * (self.steps_between_experiences + self.trajectory_steps) * 10 * 60
            if self.granularity == "M5":
                n_seconds *= 5
            if self.test:
                print("starting new stream. n minutes left:", n_seconds // 60)
            self.zeus.stream_range(start, min(start + n_seconds, int(time.time())), self.add_bar)
            start += n_seconds
            n += 1

        self.quit()

    def quit(self):
        print(("time: {time}, "
                "rewards: {reward}, "
                "temp: {actor_temp}, "
                "n exp: {n_experiences}, "
                "instr: {instrument}, "
                "start: {start}").format(
                    time=round(time.time()-self.t0, 2),
                    reward=self.total_rewards,
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
            reward_ema = float(self.server.get("test_reward_ema").decode("utf-8"))
            reward_emsd = float(self.server.get("test_reward_emsd").decode("utf-8"))
            if reward_emsd == -1:
                self.server.set("test_reward_ema", self.total_rewards)
                self.server.set("test_reward_emsd", 0)
            else:
                delta = self.total_rewards - reward_ema
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
                                       'trades_open',
                                       'spreads',
                                       'mus',
                                       'place_actions',
                                       'rewards'))
