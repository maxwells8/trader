import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import time
import heapq
import sys
sys.path.insert(0, '../worker')
from worker import Experience
import networks
from networks import *
from environment import *
import redis
import msgpack

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


class Optimizer(object):

    def __init__(self, models_loc):
        self.models_loc = models_loc
        # networks
        # this is the lstm's version
        # self.MEN = MarketEncoder().cuda()
        # this is the attention version
        self.MEN = CNNEncoder().cuda()
        self.ETO = EncoderToOthers().cuda()
        self.PN = ProbabilisticProposer().cuda()
        self.ACN = ActorCritic().cuda()
        self.ACN_ = ActorCritic().cuda()
        try:
            self.MEN.load_state_dict(torch.load(self.models_loc + 'market_encoder.pt'))
            self.ETO.load_state_dict(torch.load(self.models_loc + 'encoder_to_others.pt'))
            self.PN.load_state_dict(torch.load(self.models_loc + 'proposer.pt'))
            self.ACN.load_state_dict(torch.load(self.models_loc + 'actor_critic.pt'))
            self.ACN_.load_state_dict(torch.load(self.models_loc + 'actor_critic.pt'))
        except FileNotFoundError:
            self.MEN = CNNEncoder().cuda()
            self.ETO = EncoderToOthers().cuda()
            self.PN = ProbabilisticProposer().cuda()
            self.ACN = ActorCritic().cuda()
            self.ACN_ = ActorCritic().cuda()

            torch.save(self.MEN.state_dict(), self.models_loc + 'market_encoder.pt')
            torch.save(self.ETO.state_dict(), self.models_loc + 'encoder_to_others.pt')
            torch.save(self.PN.state_dict(), self.models_loc + 'proposer.pt')
            torch.save(self.ACN.state_dict(), self.models_loc + 'actor_critic.pt')

        self.server = redis.Redis("localhost")
        self.gamma = float(self.server.get("gamma").decode("utf-8"))
        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))
        self.max_rho = torch.Tensor([float(self.server.get("max_rho").decode("utf-8"))]).cuda()
        self.max_c = torch.Tensor([float(self.server.get("max_c").decode("utf-8"))]).cuda()

        self.proposed_target_maximization_weight = float(self.server.get("proposed_target_maximization_weight").decode("utf-8"))
        self.proposed_entropy_weight = float(self.server.get("proposed_entropy_weight").decode("utf-8"))
        self.critic_weight = float(self.server.get("critic_weight").decode("utf-8"))
        self.actor_v_weight = float(self.server.get("actor_v_weight").decode("utf-8"))
        self.actor_entropy_weight = float(self.server.get("actor_entropy_weight").decode("utf-8"))
        self.weight_penalty = float(self.server.get("weight_penalty").decode("utf-8"))

        self.learning_rate = float(self.server.get("learning_rate").decode("utf-8"))

        self.queued_batch_size = int(self.server.get("queued_batch_size").decode("utf-8"))
        self.prioritized_batch_size = int(self.server.get("prioritized_batch_size").decode("utf-8"))

        self.queued_experience = []
        self.prioritized_experience = []
        try:
            self.optimizer = optim.Adam([params for params in self.ETO.parameters()] +
                                        [params for params in self.PN.parameters()] +
                                        [params for params in self.ACN.parameters()],
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_penalty)
            checkpoint = torch.load(models_loc + "rl_train.pt")
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_step = checkpoint['steps']
            self.start_n_samples = checkpoint['n_samples']
            self.server.set("spread_reimbursement_ratio", checkpoint['spread_reimbursement_ratio'])

        except:
            self.optimizer = optim.Adam([params for params in self.ETO.parameters()] +
                                        [params for params in self.PN.parameters()] +
                                        [params for params in self.ACN.parameters()],
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_penalty)
            self.start_n_samples = 0
            self.start_step = 0
            spread_reimbursement_ratio = 0
            self.server.set("spread_reimbursement_ratio", spread_reimbursement_ratio)
            cur_state = {
                'n_samples':self.start_n_samples,
                'steps':self.start_step,
                'spread_reimbursement_ratio':spread_reimbursement_ratio,
                'optimizer':self.optimizer.state_dict()
            }
            torch.save(self.optimizer.state_dict(), self.models_loc + 'rl_train.pt')

    def run(self):
        prev_reward_ema = None
        prev_reward_emsd = None
        n_samples = self.start_n_samples
        step = self.start_step
        while True:
            t0 = time.time()
            n_experiences = 0
            # read in experience from the queue
            while True:
                if (len(self.queued_experience) < self.queued_batch_size and self.server.llen("experience") > 0):
                    experience = self.server.lpop("experience")
                    experience = msgpack.unpackb(experience, raw=False)
                    self.queued_experience.append(experience)
                    n_experiences += 1
                elif (step != 1 or (step == 1 and len(self.queued_experience) == self.queued_batch_size)) and len(self.queued_experience) + len(self.prioritized_experience) > 0:
                    break
                else:
                    experience = self.server.blpop("experience")[1]
                    experience = msgpack.unpackb(experience, raw=False)
                    self.queued_experience.append(experience)
                    n_experiences += 1

            experiences = self.queued_experience + self.prioritized_experience
            batch_size = len(experiences)

            # start grads anew
            self.optimizer.zero_grad()
            self.ACN_.zero_grad()

            # get the inputs to the networks in the right form
            batch = Experience(*zip(*experiences))
            time_states = [*zip(*batch.time_states)]
            for i, time_state_ in enumerate(time_states):
                time_states[i] = torch.Tensor(time_state_).view(batch_size, 1, networks.D_BAR)
            percent_in = [*zip(*batch.percents_in)]
            spread = [*zip(*batch.spreads)]
            mu = [*zip(*batch.mus)]
            queried_mus = [*zip(*batch.proposed_mus)]
            proposed_actions = [*zip(*batch.proposed_actions)]
            place_action = [*zip(*batch.place_actions)]
            reward = [*zip(*batch.rewards)]

            window = len(time_states) - len(percent_in)
            assert window == networks.WINDOW
            assert len(percent_in) == self.trajectory_steps

            reward_ema = float(self.server.get("reward_ema").decode("utf-8"))
            reward_emsd = float(self.server.get("reward_emsd").decode("utf-8"))

            critic_loss = torch.Tensor([0]).cuda()
            actor_v_loss = torch.Tensor([0]).cuda()
            actor_entropy_loss = torch.Tensor([0]).cuda()
            proposed_target_maximization_loss = torch.Tensor([0]).cuda()
            proposed_entropy_loss = torch.Tensor([0]).cuda()

            time_states_ = torch.cat(time_states[-window:], dim=1).detach().cuda()
            mean = time_states_[:, :, :4].contiguous().view(batch_size, -1).mean(1).view(batch_size, 1, 1)
            std = time_states_[:, :, :4].contiguous().view(batch_size, -1).std(1).view(batch_size, 1, 1)
            time_states_[:, :, :4] = (time_states_[:, :, :4] - mean) / std
            spread_ = torch.Tensor(spread[-1]).view(-1, 1, 1).cuda() / std
            time_states_ = time_states_.transpose(0, 1)

            market_encoding = self.MEN.forward(time_states_)
            market_encoding = self.ETO.forward(market_encoding, (std + 1e-9).log(), spread_, torch.Tensor(percent_in[-1]).cuda())
            queried = torch.Tensor(proposed_actions[-1]).cuda().view(batch_size, -1)
            policy, value = self.ACN.forward(market_encoding, queried)

            v_next = value
            v_trace = value
            c = 1
            t_men = 0
            t_eto = 0
            t_pn = 0
            t_acn = 0

            for i in range(1, self.trajectory_steps):
                time_states_ = torch.cat(time_states[-window-i:-i], dim=1).cuda()
                mean = time_states_[:, :, :4].contiguous().view(batch_size, -1).mean(1).view(batch_size, 1, 1)
                std = time_states_[:, :, :4].contiguous().view(batch_size, -1).std(1).view(batch_size, 1, 1)
                time_states_[:, :, :4] = (time_states_[:, :, :4] - mean) / std
                spread_ = torch.Tensor(spread[-i]).view(-1, 1, 1).cuda() / std
                time_states_ = time_states_.transpose(0, 1)

                market_encoding = self.MEN.forward(time_states_)
                market_encoding = self.ETO.forward(market_encoding, (std + 1e-9).log(), spread_, torch.Tensor(percent_in[-i-1]).cuda())
                queried = torch.Tensor(proposed_actions[-i-1]).cuda().view(batch_size, -1)
                proposed, proposed_pi, p_x_mu, p_x_sigma = self.PN.forward(market_encoding, return_params=True)
                queried_pi = self.PN.p(queried.detach(), p_x_mu, p_x_sigma)
                policy, value = self.ACN.forward(market_encoding, queried)

                pi_ = policy.gather(1, torch.Tensor(place_action[-i-1]).cuda().long().view(-1, 1))
                mu_ = torch.Tensor(mu[-i-1]).cuda().view(-1, 1)
                queried_mu = torch.Tensor(queried_mus[-i-1]).cuda().view(-1, 1)

                r = torch.Tensor(reward[-i]).cuda().view(-1, 1)

                delta_v = torch.min(self.max_rho, (pi_ / mu_) * (queried_pi / queried_mu)) * (r + self.gamma * v_next - value)
                c *= torch.min(self.max_c, (pi_ / mu_) * (queried_pi / queried_mu))

                v_trace = (value + delta_v + self.gamma * c * (v_trace - v_next)).detach()

                if i == self.trajectory_steps - 1:
                    _, target_value = self.ACN_.forward(market_encoding.detach(), proposed)
                    proposed_target_maximization_loss += (-target_value).mean()
                    proposed_entropy_loss += (proposed * torch.log(proposed + 1e-9)).mean()

                    advantage_v = torch.min(self.max_rho, (pi_/mu_) * (queried_pi / queried_mu)) * (r + self.gamma * v_trace - value)
                    actor_v_loss += (-torch.log(pi_ + 1e-9) * advantage_v.detach()).mean()

                    actor_entropy_loss += (policy * torch.log(policy + 1e-9)).mean()
                    critic_loss += F.l1_loss(value, v_trace)

                v_next = value.detach()

            total_loss = torch.Tensor([0]).cuda()
            total_loss += proposed_entropy_loss * self.proposed_entropy_weight
            total_loss += proposed_target_maximization_loss * self.proposed_target_maximization_weight
            total_loss += critic_loss * self.critic_weight
            total_loss += actor_v_loss * self.actor_v_weight
            total_loss += actor_entropy_loss * self.actor_entropy_weight

            try:
                assert torch.isnan(total_loss).sum() == 0
            except AssertionError:
                print("proposed_entropy_loss", proposed_entropy_loss)
                print("proposed_target_maximization_loss", proposed_target_maximization_loss)
                print("critic_loss", critic_loss)
                print("actor_v_loss", actor_v_loss)
                print("actor_entropy_loss", actor_entropy_loss)
                raise AssertionError("total loss is not 0")
            total_loss.backward()
            self.optimizer.step()

            step += 1
            n_samples += len(self.queued_experience)

            prev_reward_ema = reward_ema
            prev_reward_emsd = reward_emsd

            try:
                torch.save(self.ETO.state_dict(), self.models_loc + 'encoder_to_others.pt')
                torch.save(self.PN.state_dict(), self.models_loc + "proposer.pt")
                torch.save(self.ACN.state_dict(), self.models_loc + "actor_critic.pt")
                cur_state = {
                    'n_samples':n_samples,
                    'steps':step,
                    'spread_reimbursement_ratio':self.server.get("spread_reimbursement_ratio"),
                    'optimizer':self.optimizer.state_dict()
                }
                torch.save(cur_state, self.models_loc + 'rl_train.pt')
                self.ACN_.load_state_dict(torch.load(self.models_loc + 'actor_critic.pt'))
            except Exception:
                print("failed to save")

            dif = value.detach() - v_trace.detach()
            if batch_size > len(self.queued_experience):
                smallest, i_smallest = torch.min(torch.abs(dif[len(self.queued_experience):]), dim=0)
            for i, experience in enumerate(self.queued_experience):
                if len(self.prioritized_experience) == self.prioritized_batch_size and self.prioritized_batch_size != 0 and len(self.queued_experience) != batch_size:
                    if torch.abs(dif[i]) > smallest:
                        self.prioritized_experience[i_smallest] = experience
                        dif[len(self.queued_experience) + i_smallest] = torch.abs(dif[i])
                        smallest, i_smallest = torch.min(torch.abs(dif[len(self.queued_experience):]), dim=0)
                elif len(self.prioritized_experience) < self.prioritized_batch_size:
                    self.prioritized_experience.append(experience)

            self.queued_experience = []
            torch.cuda.empty_cache()

            print('-----------------------------------------------------------')
            print("n samples: {n}, batch size: {b}, steps: {s}, time: {t}".format(n=n_samples, b=batch_size, s=step, t=round(time.time()-t0, 5)))
            print()

            print("policy[0] min mean max:\n", round(policy[:, 0].cpu().detach().min().item(), 7), round(policy[:, 0].cpu().detach().mean().item(), 7), round(policy[:, 0].cpu().detach().max().item(), 7))
            print("policy[1] min mean max:\n", round(policy[:, 1].cpu().detach().min().item(), 7), round(policy[:, 1].cpu().detach().mean().item(), 7), round(policy[:, 1].cpu().detach().max().item(), 7))
            print("policy[2] min mean max:\n", round(policy[:, 2].cpu().detach().min().item(), 7), round(policy[:, 2].cpu().detach().mean().item(), 7), round(policy[:, 2].cpu().detach().max().item(), 7))
            print()

            print("proposed[0] min mean max:\n", round(proposed[:, 0].cpu().detach().min().item(), 7), round(proposed[:, 0].cpu().detach().mean().item(), 7), round(proposed[:, 0].cpu().detach().max().item(), 7))
            print("proposed[1] min mean max:\n", round(proposed[:, 1].cpu().detach().min().item(), 7), round(proposed[:, 1].cpu().detach().mean().item(), 7), round(proposed[:, 1].cpu().detach().max().item(), 7))
            print()

            print("proposed probabilities min mean max:\n", round(proposed_pi.cpu().detach().min().item(), 7), round(proposed_pi.cpu().detach().mean().item(), 7), round(proposed_pi.cpu().detach().max().item(), 7))
            print("queried probabilities min mean max:\n", round(queried_pi.cpu().detach().min().item(), 7), round(queried_pi.cpu().detach().mean().item(), 7), round(queried_pi.cpu().detach().max().item(), 7))
            print()

            queried_actions = torch.Tensor(proposed_actions).squeeze()
            print("queried[0] min mean max:\n", round(queried_actions[:, :, 0].cpu().detach().min().item(), 7), round(queried_actions[:, :, 0].cpu().detach().mean().item(), 7), round(queried_actions[:, :, 0].cpu().detach().max().item(), 7))
            print("queried[1] min mean max:\n", round(queried_actions[:, :, 1].cpu().detach().min().item(), 7), round(queried_actions[:, :, 1].cpu().detach().mean().item(), 7), round(queried_actions[:, :, 1].cpu().detach().max().item(), 7))
            print()

            print("target value sample:\n", v_trace[:4].cpu().detach().numpy())
            print("guessed value sample:\n", value[:4].cpu().detach().numpy())
            print()

            print("weighted critic loss:", round(float(critic_loss * self.critic_weight), 5))
            print('-----------------------------------------------------------')
