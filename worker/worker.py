import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
from networks import *
from environment import *
import redis
import pickle

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)

"""
the worker acts in the environment with a fixed policy for a certain amount of
time, and it also prepares the experience for the optimizer.
"""
class Worker(object):

    def __init__(self, source, name, models_loc, window, start, n_steps, test=False):

        self.environment = Env(source, start, n_steps, time_window=window)
        while True:
            # putting this while loop here because sometime the optimizer
            # is writing to the files and causes an exception
            try:
                self.market_encoder = MarketEncoder()
                self.proposer = Proposer()
                self.actor_critic = ActorCritic()
                self.market_encoder.load_state_dict(torch.load(models_loc + '/market_encoder.pt'))
                self.proposer.load_state_dict(torch.load(models_loc + '/proposer.pt'))
                self.actor_critic.load_state_dict(torch.load(models_loc + '/actor_critic.pt'))
                self.market_encoder = self.market_encoder.cpu()
                self.proposer = self.proposer.cpu()
                self.actor_critic = self.actor_critic.cpu()
                break
            except Exception:
                pass
        self.models_loc = models_loc

        self.server = redis.Redis("localhost")
        self.name = name
        self.proposed_sigma = float(self.server.get("proposed_sigma_" + name).decode("utf-8"))
        self.policy_sigma = float(self.server.get("policy_sigma_" + name).decode("utf-8"))

        self.window = window
        self.n_steps = n_steps

        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.test = test

    def run(self):
        all_rewards = []

        time_states = []
        percents_in = []
        proposed_actions = []
        mus = []
        actions = []
        rewards = []

        state = self.environment.get_state()
        initial_time_states, initial_percent_in, _ = state
        initial_time_states = torch.cat(initial_time_states).cpu()
        mean = initial_time_states[:, 0, :4].mean()
        std = initial_time_states[:, 0, :4].std()
        initial_time_states[:, 0, :4] = (initial_time_states[:, 0, :4] - mean) / std
        initial_time_states[:, 0, 4] = initial_time_states[:, 0, 4] / std
        t0 = time.time()
        for i_step in range(self.n_steps):

            if i_step >= self.window:
                time_states.append(initial_time_states)
                percents_in.append(initial_percent_in)

                market_encoding = self.market_encoder.forward(initial_time_states, torch.Tensor([initial_percent_in]).cpu(), 'cpu')

                queried_actions = self.proposer.forward(market_encoding, torch.randn(1, 2).cpu() * self.proposed_sigma)
                proposed_actions.append(queried_actions)

                policy, value = self.actor_critic.forward(market_encoding, queried_actions, self.policy_sigma)

                if self.test:
                    print("step:", i_step)
                    print("queried_actions:", queried_actions)
                    print("(policy, value):", policy, value)
                    print("rewards:", np.sum(all_rewards))

                action = int(torch.multinomial(policy, 1))
                mu = policy[0, action]
                actions.append(action)
                mus.append(mu)

                if action == 0:
                    placed_order = [0, float(queried_actions[0, 0])]
                elif action == 1:
                    placed_order = [1, float(queried_actions[0, 1])]
                else:
                    placed_order = [action]

                self.environment.step(placed_order)

            else:
                time_states.append(None)
                percents_in.append(None)
                proposed_actions.append(None)
                actions.append(None)
                mus.append(None)
                self.environment.step([3])

            state = self.environment.get_state()
            if not state:
                break

            final_time_states, final_percent_in, reward = state
            reward *= 1000
            final_time_states = torch.cat(final_time_states).cpu()
            mean = final_time_states[:, 0, :4].mean()
            std = final_time_states[:, 0, :4].std()
            final_time_states[:, 0, :4] = (final_time_states[:, 0, :4] - mean) / std
            final_time_states[:, 0, 4] = final_time_states[:, 0, 4] / std

            rewards.append(reward)
            all_rewards.append(reward)

            if i_step >= self.trajectory_steps + self.window - 1:
                experience = Experience(time_states + [final_time_states], percents_in + [final_percent_in], mus, proposed_actions, actions, rewards)
                if not self.test:
                    self.server.rpush("experience", pickle.dumps(experience, protocol=pickle.HIGHEST_PROTOCOL))

            initial_time_states = final_time_states
            initial_percent_in = final_percent_in

            if len(time_states) == self.trajectory_steps:
                del time_states[0]
                del percents_in[0]
                del mus[0]
                del proposed_actions[0]
                del actions[0]
                del rewards[0]

        print("name: {name}, steps: {steps}, time: {time}, sum all rewards = {reward}".format(name=self.name, steps=i_step, time=time.time()-t0, reward=np.sum(all_rewards)))


Experience = namedtuple('Experience', ('time_states',
                                       'percents_in',
                                       'mus',
                                       'proposed_actions',
                                       'place_actions',
                                       'rewards'))

if __name__ == "__main__":
    server = redis.Redis("localhost")
    server.set("proposed_sigma_test", 0)
    server.set("policy_sigma_test", 1)
    source = "C:\\Users\\Preston\\Programming\\trader\\normalized_data\\DAT_MT_EURUSD_M1_2017-1.1294884577273274.csv"
    models_loc = '../models'
    window = 256
    n_steps = 100000
    test = True
    np.random.seed(int(time.time()))
    worker = Worker(source, "test", models_loc, window, np.random.randint(0,200000), n_steps, test)
    worker.run()
