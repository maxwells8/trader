import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.insert(0, '../worker')
from worker import Experience
from networks import *
from environment import *
import redis
import pickle

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

"""
TODO:
- either make this work with variable time_state lengths, or don't let
them come into the experience
"""
class Optimizer(object):

    def __init__(self, models_loc):
        self.models_loc = models_loc
        # networks
        self.MEN = torch.load(self.models_loc + '/market_encoder.pt').cuda()
        self.PN = torch.load(self.models_loc + '/proposer.pt').cuda()
        self.ACN = torch.load(self.models_loc + '/actor_critic.pt').cuda()

        # target networks
        self.MEN_ = torch.load(self.models_loc + '/market_encoder.pt').cuda()
        self.ACN_ = torch.load(self.models_loc + '/actor_critic.pt').cuda()

        self.server = redis.Redis("localhost")
        self.gamma = float(self.server.get("gamma").decode("utf-8"))
        self.tau = float(self.server.get("optimizer_tau").decode("utf-8"))
        self.max_rho = torch.Tensor([float(self.server.get("optimizer_max_rho").decode("utf-8"))], device='cuda')

        self.proposed_weight = float(self.server.get("optimizer_proposed_weight").decode("utf-8"))
        self.critic_weight = float(self.server.get("optimizer_critic_weight").decode("utf-8"))
        self.actor_weight = float(self.server.get("optimizer_actor_weight").decode("utf-8"))
        self.entropy_weight = float(self.server.get("optimizer_entropy_weight").decode("utf-8"))
        self.weight_penalty = float(self.server.get("optimizer_weight_penalty").decode("utf-8"))

        self.batch_size = int(self.server.get("optimizer_batch_size").decode("utf-8"))

        self.experience = []
        self.optimizer = optim.Adam([params for params in self.MEN.parameters()] +
                                    [params for params in self.PN.parameters()] +
                                    [params for params in self.ACN.parameters()],
                                    weight_decay=self.weight_penalty)


    def run(self):

        while len(self.experience) < self.batch_size:
            experience = pickle.loads(self.server.lpop("experience"))
            self.experience.append(experience)

        while True:
            # start grads anew
            self.optimizer.zero_grad()

            # get the inputs to the networks in the right form
            batch = Experience(*zip(*self.experience))  # maybe restructure this so that we aren't using the entire experience each batch
            initial_time_states = torch.cat(batch.initial_time_states, dim=1).cuda()
            initial_percent_in = torch.Tensor(batch.initial_percent_in).resize(self.batch_size, 1).cuda()
            mu = torch.Tensor(batch.mu).resize(self.batch_size, 1).cuda()
            proposed = torch.cat(batch.proposed_actions).resize(self.batch_size, 2).cuda()
            place_action = torch.Tensor(batch.place_action).resize(self.batch_size, 1).cuda()
            reward = torch.Tensor(batch.reward).resize(self.batch_size, 1).cuda()
            final_time_states = torch.cat(batch.final_time_states, dim=1).cuda()
            final_percent_in = torch.Tensor(batch.final_percent_in).resize(self.batch_size, 1).cuda()

            # calculate the market_encoding
            initial_market_encoding = self.MEN.forward(initial_time_states, initial_percent_in, 'cuda')
            final_market_encoding = self.MEN_.forward(final_time_states, final_percent_in, 'cuda')

            # proposed loss
            proposed_actions = self.PN.forward(initial_market_encoding)
            _, target_value = self.ACN_.forward(initial_market_encoding, proposed_actions)
            (self.proposed_weight * -target_value.mean()).backward()

            # critic loss
            expected_policy, expected_value = self.ACN.forward(initial_market_encoding, proposed)
            this_step_policy, this_step_value = self.ACN_.forward(initial_market_encoding, proposed)
            next_step_policy, next_step_value = self.ACN_.forward(final_market_encoding, proposed_actions)
            delta_V = torch.min(self.max_rho, this_step_policy.gather(1, place_action.view(-1, 1))/mu)*(reward + (next_step_value * self.gamma) - this_step_value)
            v = this_step_value + delta_V

            critic_loss = F.smooth_l1_loss(expected_value, target_value)
            (self.critic_weight * critic_loss).backward()

            # policy loss
            policy_loss = -torch.log(expected_policy.gather(1, place_action.view(-1, 1))) * delta_V
            (self.actor_weight * policy_loss.mean()).backward()

            # entropy
            entropy_loss = expected_policy * torch.log(expected_policy)
            (self.entropy_weight * entropy_loss.mean()).backward()

            # take a step
            self.optimizer.step()

            # update the target networks using a exponential moving average
            for i, param in enumerate(self.MEN_.parameters()):
                param = (self.tau * self.MEN.parameters()[i]) + ((1 - self.tau) * param)

            for i, param in enumerate(self.ACN_.parameters()):
                param = (self.tau * self.ACN.parameters()[i]) + ((1 - self.tau) * param)

            if int(self.server.get("optimizer_update").decode("utf-8")):
                self.experience = pickle.loads(self.server.get("experience"))
                self.MEN.save(self.models_loc + "/market_encoder.pt")
                self.PN.save(self.models_loc + "/proposer.pt")
                self.ACN.save(self.models_loc + "/actor_critic.pt")
