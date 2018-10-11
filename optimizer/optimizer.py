import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import heapq
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
        self.MEN = MarketEncoder().cuda()
        self.PN = Proposer().cuda()
        self.ACN = ActorCritic().cuda()
        self.ACN_ = ActorCritic().cuda()
        try:
            self.MEN.load_state_dict(torch.load(self.models_loc + '/market_encoder.pt'))
            self.PN.load_state_dict(torch.load(self.models_loc + '/proposer.pt'))
            self.ACN.load_state_dict(torch.load(self.models_loc + '/actor_critic.pt'))
            self.ACN_.load_state_dict(torch.load(self.models_loc + '/actor_critic.pt'))
        except Exception:
            self.MEN = MarketEncoder().cuda()
            self.PN = Proposer().cuda()
            self.ACN = ActorCritic().cuda()
            self.ACN_ = ActorCritic().cuda()

            torch.save(self.MEN.state_dict(), self.models_loc + '/market_encoder.pt')
            torch.save(self.PN.state_dict(), self.models_loc + '/proposer.pt')
            torch.save(self.ACN.state_dict(), self.models_loc + '/actor_critic.pt')

        self.server = redis.Redis("localhost")
        self.gamma = float(self.server.get("gamma").decode("utf-8"))
        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))
        self.tau = float(self.server.get("optimizer_tau").decode("utf-8"))
        self.max_rho = torch.Tensor([float(self.server.get("optimizer_max_rho").decode("utf-8"))], device='cuda')
        self.max_c = torch.Tensor([float(self.server.get("optimizer_max_c").decode("utf-8"))], device='cuda')

        self.proposed_weight = float(self.server.get("optimizer_proposed_weight").decode("utf-8"))
        self.critic_weight = float(self.server.get("optimizer_critic_weight").decode("utf-8"))
        self.actor_weight = float(self.server.get("optimizer_actor_weight").decode("utf-8"))
        self.entropy_weight = float(self.server.get("optimizer_entropy_weight").decode("utf-8"))
        self.weight_penalty = float(self.server.get("optimizer_weight_penalty").decode("utf-8"))
        self.learning_rate = float(self.server.get("optimizer_learning_rate").decode("utf-8"))

        self.queued_batch_size = int(self.server.get("optimizer_queued_batch_size").decode("utf-8"))
        self.prioritized_batch_size = int(self.server.get("optimizer_prioritized_batch_size").decode("utf-8"))

        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.queued_experience = []
        self.prioritized_experience = []
        try:
            self.optimizer = optim.Adam([params for params in self.MEN.parameters()] +
                                        [params for params in self.PN.parameters()] +
                                        [params for params in self.ACN.parameters()],
                                        weight_decay=self.weight_penalty)
            self.optimizer.load_state_dict(torch.load(models_loc + "/optimizer.pt"))

        except:
            self.optimizer = optim.Adam([params for params in self.MEN.parameters()] +
                                        [params for params in self.PN.parameters()] +
                                        [params for params in self.ACN.parameters()],
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_penalty)
            torch.save(self.optimizer.state_dict(), self.models_loc + '/optimizer.pt')

    def run(self):

        n_experiences = 0
        step = 1
        while True:
            # read in experience from the queue
            while True:
                if (len(self.queued_experience) < self.queued_batch_size and self.server.llen("experience") > 0):
                    experience = self.server.lpop("experience")
                    experience = pickle.loads(experience)
                    self.queued_experience.append(experience)
                    n_experiences += 1
                elif step != 1 or (step == 1 and len(self.queued_experience) == self.queued_batch_size) or len(self.queued_experience) + len(self.prioritized_experience) > 0:
                    break
                else:
                    experience = self.server.blpop("experience")[1]
                    experience = pickle.loads(experience)
                    self.queued_experience.append(experience)
                    n_experiences += 1

            experiences = self.queued_experience + self.prioritized_experience

            # start grads anew
            self.optimizer.zero_grad()
            self.ACN_.zero_grad()

            # get the inputs to the networks in the right form
            batch = Experience(*zip(*experiences))
            time_states = [*zip(*batch.time_states)]
            percent_in = [*zip(*batch.percents_in)]
            mu = [*zip(*batch.mus)]
            proposed_actions = [*zip(*batch.proposed_actions)]
            place_action = [*zip(*batch.place_actions)]
            reward = [*zip(*batch.rewards)]

            c = 1
            market_encoding = self.MEN.forward(torch.cat(time_states[0], dim=1).detach().cuda(), torch.Tensor(percent_in[0]), 'cuda')
            proposed = self.PN.forward(market_encoding)
            initial_policy, initial_value = self.ACN(market_encoding, proposed)
            _, target_value = self.ACN_(market_encoding, torch.cat(proposed_actions[0]).cuda())
            proposed_loss = (-target_value).mean()
            policy = initial_policy.clone()
            value = initial_value.clone()
            v_trace = value
            for i in range(self.trajectory_steps - 1):
                market_encoding = self.MEN.forward(torch.cat(time_states[i+1], dim=1).detach().cuda(), torch.Tensor(percent_in[i]), 'cuda')
                proposed = self.PN.forward(market_encoding)
                next_policy, next_value = self.ACN(market_encoding, proposed)
                delta_V = torch.min(self.max_rho, policy.gather(1, torch.Tensor(place_action[i]).long().view(-1, 1))/torch.Tensor(mu[i]).view(-1, 1))
                delta_V *= (torch.Tensor(reward[i]).view(-1, 1) + self.gamma * next_value - value)
                v_trace += c * (self.gamma ** i) * delta_V.detach()
                if i == 0:
                    second_delta_V = delta_V.clone()

                c *= torch.Tensor(mu[i]).view(-1, 1)
                value = next_value.clone()
                policy = next_policy.clone()

            critic_loss = F.mse_loss(initial_value, v_trace)
            actor_loss = -torch.max(torch.Tensor([-10]), torch.log(initial_policy.gather(1, torch.Tensor(place_action[0]).long().view(-1, 1))))
            actor_loss *= torch.min(self.max_rho, policy.gather(1, torch.Tensor(place_action[0]).long().view(-1, 1))/torch.Tensor(mu[0]).view(-1, 1))
            actor_loss *= second_delta_V
            actor_loss = actor_loss.mean()
            entropy_loss = (initial_policy * torch.max(torch.Tensor([-10]), torch.log(initial_policy))).mean()

            # initial_time_states = torch.cat(batch.initial_time_states, dim=1).detach().cuda()
            # initial_percent_in = torch.Tensor(batch.initial_percent_in).resize(len(experiences), 1).cuda()
            # mu = torch.Tensor(batch.mu).resize(len(experiences), 1).cuda()
            # proposed = torch.cat(batch.proposed_actions).resize(len(experiences), 2).detach().cuda()
            # place_action = torch.Tensor(batch.place_action).long().resize(len(experiences), 1).cuda()
            # reward = torch.Tensor(batch.reward).resize(len(experiences), 1).cuda()
            # final_time_states = torch.cat(batch.final_time_states, dim=1).detach().cuda()
            # final_percent_in = torch.Tensor(batch.final_percent_in).resize(len(experiences), 1).cuda()

            # calculate the market_encoding
            # initial_market_encoding = self.MEN.forward(initial_time_states, initial_percent_in, 'cuda')
            # initial_market_encoding_ = self.MEN_.forward(initial_time_states, initial_percent_in, 'cuda').detach()
            # final_market_encoding = self.MEN_.forward(final_time_states, final_percent_in, 'cuda')
            #
            # # proposed loss
            # proposed_actions = self.PN.forward(initial_market_encoding)
            # _, target_value = self.ACN_.forward(initial_market_encoding_, proposed_actions)
            # proposed_loss = -target_value.mean()
            #
            # # critic loss
            # expected_policy, expected_value = self.ACN.forward(initial_market_encoding, proposed)
            # this_step_policy, this_step_value = self.ACN_.forward(initial_market_encoding_, proposed)
            # this_step_policy = this_step_policy.detach()
            # this_step_value = this_step_value.detach()
            # next_step_proposed = self.PN_.forward(final_market_encoding).detach()
            # next_step_policy, next_step_value = self.ACN_.forward(final_market_encoding, next_step_proposed)
            # next_step_policy = next_step_policy.detach()
            # next_step_value = next_step_value.detach()
            # delta_V = (torch.min(self.max_rho, this_step_policy.gather(1, place_action.view(-1, 1))/mu)*(reward + (next_step_value * (self.gamma**(self.trajectory_steps+1))) - this_step_value)).detach()
            #
            # v = this_step_value + delta_V
            #
            # critic_loss = F.l1_loss(expected_value, v)
            #
            # # policy loss
            # policy_loss = (-torch.max(torch.Tensor([-10]), torch.log(expected_policy.gather(1, place_action.view(-1, 1)))) * delta_V).mean()
            #
            # # policy entropy
            # entropy_loss = (expected_policy * torch.max(torch.Tensor([-10]), torch.log(expected_policy))).mean()

            total_loss = proposed_loss * self.proposed_weight
            total_loss += critic_loss * self.critic_weight
            total_loss += actor_loss * self.actor_weight
            total_loss += entropy_loss * self.entropy_weight

            total_loss.backward()
            self.optimizer.step()

            if step % 10 == 0:
                print("n experiences: {n}, steps: {s}, loss: {l}".format(n=n_experiences, s=step, l=total_loss))
                try:
                    torch.save(self.MEN.state_dict(), self.models_loc + "/market_encoder.pt")
                    torch.save(self.PN.state_dict(), self.models_loc + "/proposer.pt")
                    torch.save(self.ACN.state_dict(), self.models_loc + "/actor_critic.pt")
                    torch.save(self.optimizer.state_dict(), self.models_loc + "/optimizer.pt")
                    self.ACN_.load_state_dict(torch.load(self.models_loc + '/actor_critic.pt'))
                except Exception:
                    print("failed to save")

            for i, experience in enumerate(self.queued_experience):
                if len(self.prioritized_experience) == self.prioritized_batch_size and self.prioritized_batch_size != 0:
                    smallest, i_smallest = torch.min(torch.abs(delta_V[len(self.queued_experience):]), dim=0)
                    if torch.abs(delta_V[i]) > smallest:
                        self.prioritized_experience[i_smallest] = experience
                        delta_V[len(self.queued_experience) + i_smallest] = torch.abs(delta_V[i])
                elif self.prioritized_batch_size != 0:
                    self.prioritized_experience.append(experience)

            self.queued_experience = []
            step += 1
