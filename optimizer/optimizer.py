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
import os

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


class Optimizer(object):

    def __init__(self, models_loc):
        self.models_loc = models_loc
        self.server = redis.Redis("localhost")
        self.MEN = LSTMEncoder().cuda()
        self.ETO = EncoderToOthers().cuda()
        self.ACN = ActorCritic().cuda()
        try:
            # MEN_compressed_state_dict = self.server.get("market_encoder")
            # ETO_compressed_state_dict = self.server.get("encoder_to_others")
            # ACN_compressed_state_dict = self.server.get("actor_critic")
            #
            # MEN_state_dict = msgpack.unpackb(MEN_compressed_state_dict, raw=False)
            # ETO_state_dict = msgpack.unpackb(ETO_compressed_state_dict, raw=False)
            # ACN_state_dict = msgpack.unpackb(ACN_compressed_state_dict, raw=False)
            #
            # self.MEN.load_state_dict(MEN_state_dict)
            # self.ETO.load_state_dict(ETO_state_dict)
            # self.ACN.load_state_dict(ACN_state_dict)

            self.MEN.load_state_dict(torch.load(self.models_loc + 'market_encoder.pt'))
            self.ETO.load_state_dict(torch.load(self.models_loc + 'encoder_to_others.pt'))
            self.ACN.load_state_dict(torch.load(self.models_loc + 'actor_critic.pt'))

        except FileNotFoundError:
            self.MEN = LSTMEncoder().cuda()
            self.ETO = EncoderToOthers().cuda()
            self.ACN = ActorCritic().cuda()

            # MEN_state_dict = msgpack.packb(self.MEN.state_dict(), use_bin_type=True)
            # ETO_state_dict = msgpack.packb(self.ETO.state_dict(), use_bin_type=True)
            # ACN_state_dict = msgpack.packb(self.ACN.state_dict(), use_bin_type=True)
            #
            # self.server.set("market_encoder", MEN_state_dict)
            # self.server.set("encoder_to_others", MEN_state_dict)
            # self.server.set("actor_critic", MEN_state_dict)

            torch.save(self.MEN.state_dict(), self.models_loc + 'market_encoder.pt')
            torch.save(self.ETO.state_dict(), self.models_loc + 'encoder_to_others.pt')
            torch.save(self.ACN.state_dict(), self.models_loc + 'actor_critic.pt')

        self.actor_temp_cooldown = float(self.server.get("actor_temp_cooldown").decode("utf-8"))
        self.gamma = float(self.server.get("gamma").decode("utf-8"))
        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))
        self.max_rho = torch.Tensor([float(self.server.get("max_rho").decode("utf-8"))]).cuda()
        self.max_c = torch.Tensor([float(self.server.get("max_c").decode("utf-8"))]).cuda()

        self.critic_weight = float(self.server.get("critic_weight").decode("utf-8"))
        self.actor_v_weight = float(self.server.get("actor_v_weight").decode("utf-8"))
        self.actor_entropy_weight = float(self.server.get("actor_entropy_weight").decode("utf-8"))
        self.weight_penalty = float(self.server.get("weight_penalty").decode("utf-8"))

        self.learning_rate = float(self.server.get("learning_rate").decode("utf-8"))

        self.queued_batch_size = int(self.server.get("queued_batch_size").decode("utf-8"))

        self.queued_experience = []
        self.prioritized_experience = []
        try:
            self.optimizer = optim.Adam([param for param in self.MEN.parameters()] +
                                        [param for param in self.ETO.parameters()] +
                                        [param for param in self.ACN.parameters()],
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_penalty)

            checkpoint = torch.load(models_loc + "rl_train.pt")
            # checkpoint_compressed_dict = self.server.get("optimizer")
            # checkpoint = msgpack.unpackb(checkpoint_compressed_dict, raw=False)

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_step = checkpoint['steps']
            self.start_n_samples = checkpoint['n_samples']
            self.actor_temp = checkpoint['actor_temp']

        except:
            self.optimizer = optim.Adam([param for param in self.MEN.parameters()] +
                                        [param for param in self.ETO.parameters()] +
                                        [param for param in self.ACN.parameters()],
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_penalty)
            self.start_step = 0
            self.start_n_samples = 0
            self.actor_temp = 5
            cur_state = {
                'n_samples':self.start_n_samples,
                'steps':self.start_step,
                'actor_temp':self.actor_temp,
                'optimizer':self.optimizer.state_dict()
            }
            torch.save(cur_state, self.models_loc + 'rl_train.pt')

            cur_state_compressed = msgpack.packb(cur_state, use_bin_type=True)
            self.server.set("optimizer", cur_state_compressed)

        self.original_actor_temp = 5
        self.server.set("actor_temp", self.actor_temp)

    def run(self):
        self.MEN.train()
        self.ETO.train()
        self.ACN.train()

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
                elif (step != 0 or (step == 0 and len(self.queued_experience) == self.queued_batch_size)) and len(self.queued_experience) > 1:
                    break
                else:
                    experience = self.server.blpop("experience")[1]
                    experience = msgpack.unpackb(experience, raw=False)
                    self.queued_experience.append(experience)
                    n_experiences += 1

            # get some experiences from the replay buffer
            buffer_size = min(self.server.llen("replay_buffer"), int(self.server.get("replay_buffer_size").decode("utf-8")))
            n_replay = 0
            if buffer_size > len(self.queued_experience):
                while n_replay < len(self.queued_experience):
                    try:
                        buffer_size = min(self.server.llen("replay_buffer"), int(self.server.get("replay_buffer_size").decode("utf-8")))
                        loc = np.random.randint(0, buffer_size)
                        experience = self.server.lindex("replay_buffer", int(loc))
                        experience = msgpack.unpackb(experience, raw=False)
                        self.prioritized_experience.append(experience)
                        n_replay += 1
                    except Exception:
                        pass

            experiences = self.queued_experience + self.prioritized_experience
            batch_size = len(experiences)

            # if step in [10000]:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] / 10

            # start grads anew
            self.optimizer.zero_grad()

            # get the inputs to the networks in the right form
            batch = Experience(*zip(*experiences))
            time_states = [*zip(*batch.time_states)]
            for i, time_state_ in enumerate(time_states):
                time_states[i] = torch.Tensor(time_state_).view(batch_size, 1, networks.D_BAR)
            percent_in = [*zip(*batch.percents_in)]
            spread = [*zip(*batch.spreads)]
            mu = [*zip(*batch.mus)]
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

            # print("time_states[{i}][-1]".format(i=self.trajectory_steps), time_states[-1])
            # print("percent_in[{i}]".format(i=self.trajectory_steps-1), percent_in[self.trajectory_steps-1])
            # print("spread[{i}]".format(i=self.trajectory_steps-1), spread[self.trajectory_steps-1])
            # print()
            time_states_ = torch.cat(time_states[self.trajectory_steps:], dim=1).detach().cuda()
            mean = time_states_[:, :, :4].contiguous().view(batch_size, 4 * window).mean(1).view(batch_size, 1, 1)
            std = time_states_[:, :, :4].contiguous().view(batch_size, 4 * window).std(1).view(batch_size, 1, 1)
            time_states_[:, :, :4] = (time_states_[:, :, :4] - mean) / std
            spread_ = torch.Tensor(spread[self.trajectory_steps-1]).view(-1, 1, 1).cuda() / std
            time_states_ = time_states_.transpose(0, 1)

            market_encoding = self.MEN.forward(time_states_)
            market_encoding = self.ETO.forward(market_encoding, spread_, torch.Tensor(percent_in[self.trajectory_steps-1]).cuda())
            policy, value = self.ACN.forward(market_encoding)

            v_next = value.detach()
            v_trace = value.detach()
            for i in range(self.trajectory_steps - 2, -1, -1):
                # print(i)
                # print("time_states[{i}][-1]".format(i=i+1), time_states[window+i])
                # print("percent_in[{i}]".format(i=i), percent_in[i])
                # print("spread[{i}]".format(i=i), spread[i])
                # print("place_action[{i}]".format(i=i), place_action[i])
                # print("mu[{i}]".format(i=i), mu[i])
                # print("reward[{i}]".format(i=i), reward[i])
                # print()

                time_states_ = torch.cat(time_states[i+1:window+i+1], dim=1).cuda()
                mean = time_states_[:, :, :4].contiguous().view(batch_size, 4 * window).mean(1).view(batch_size, 1, 1)
                std = time_states_[:, :, :4].contiguous().view(batch_size, 4 * window).std(1).view(batch_size, 1, 1)
                time_states_[:, :, :4] = (time_states_[:, :, :4] - mean) / std
                spread_ = torch.Tensor(spread[i]).view(-1, 1, 1).cuda() / std
                time_states_ = time_states_.transpose(0, 1)

                market_encoding = self.MEN.forward(time_states_)
                market_encoding = self.ETO.forward(market_encoding, spread_, torch.Tensor(percent_in[i]).cuda())
                policy, value = self.ACN.forward(market_encoding)

                pi_ = policy.gather(1, torch.Tensor(place_action[i]).cuda().long().view(batch_size, 1))
                mu_ = torch.Tensor(mu[i]).cuda().view(batch_size, 1)

                r = torch.Tensor(reward[i]).cuda().view(batch_size, 1)

                if i == 0:
                    rho = torch.min(self.max_rho, pi_ / (mu_ + 1e-9))
                    # rho = torch.ones_like(rho)

                    advantage_v = r + self.gamma * v_trace - value
                    actor_v_loss += (-torch.log(pi_ + 1e-9) * (rho * advantage_v).detach()).mean()

                    # q = r + self.gamma * v_trace
                    # actor_v_loss += (-torch.log(pi_ + 1e-9) * (rho * q).detach()).mean()

                    actor_entropy_loss += (torch.log(policy + 1e-9) * policy).mean()

                rho = torch.min(self.max_rho, pi_ / (mu_ + 1e-9))
                # rho = torch.ones_like(rho)

                c = torch.min(self.max_c, pi_ / (mu_ + 1e-9))
                # c = torch.ones_like(c)
                delta_v = rho * (r + self.gamma * v_next - value)

                v_trace = (value + delta_v + self.gamma * c * (v_trace - v_next)).detach()

                if i == 0:
                    critic_loss += nn.MSELoss()(value, v_trace.detach())

                v_next = value.detach()

            total_loss = torch.Tensor([0]).cuda()
            total_loss += critic_loss * self.critic_weight
            total_loss += actor_v_loss * self.actor_v_weight
            total_loss += actor_entropy_loss * self.actor_entropy_weight

            try:
                assert torch.isnan(total_loss).sum() == 0
            except AssertionError:
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
                torch.save(self.MEN.state_dict(), self.models_loc + 'market_encoder.pt')
                torch.save(self.ETO.state_dict(), self.models_loc + 'encoder_to_others.pt')
                torch.save(self.ACN.state_dict(), self.models_loc + "actor_critic.pt")
                cur_state = {
                    'n_samples':n_samples,
                    'steps':step,
                    'actor_temp':self.actor_temp,
                    'optimizer':self.optimizer.state_dict()
                }
                torch.save(cur_state, self.models_loc + 'rl_train.pt')
            except Exception:
                print("failed to save")

            if step % 10000 == 0:
                try:
                    if not os.path.exists(self.models_loc + 'model_history'):
                        os.makedirs(self.models_loc + 'model_history')
                    if not os.path.exists(self.models_loc + 'model_history/{step}'.format(step=step)):
                        os.makedirs(self.models_loc + 'model_history/{step}'.format(step=step))
                    torch.save(self.MEN.state_dict(), self.models_loc + 'model_history/{step}/market_encoder.pt'.format(step=step))
                    torch.save(self.ETO.state_dict(), self.models_loc + 'model_history/{step}/encoder_to_others.pt'.format(step=step))
                    torch.save(self.ACN.state_dict(), self.models_loc + "model_history/{step}/actor_critic.pt".format(step=step))
                    cur_state = {
                        'n_samples':n_samples,
                        'steps':step,
                        'actor_temp':self.actor_temp,
                        'optimizer':self.optimizer.state_dict()
                    }
                    torch.save(cur_state, self.models_loc + 'model_history/{step}/rl_train.pt'.format(step=step))
                except Exception:
                    print("failed to save")

            self.actor_temp = 1 + (self.original_actor_temp - 1) * self.actor_temp_cooldown ** step
            self.server.set("actor_temp", self.actor_temp)
            self.queued_experience = []
            self.prioritized_experience = []
            # torch.cuda.empty_cache()

            print('-----------------------------------------------------------')
            print("n samples: {n}, batch size: {b}, steps: {s}, time: {t}".format(n=n_samples, b=batch_size, s=step, t=round(time.time()-t0, 5)))
            print()

            print("policy means:", policy.cpu().detach().mean(dim=0))
            print()

            print("value min mean std max:\n", round(value.cpu().detach().min().item(), 7), round(value.cpu().detach().mean().item(), 7), round(value.cpu().detach().std().item(), 7), round(value.cpu().detach().max().item(), 7))
            print("v_trace min mean std max:\n", round(v_trace.cpu().detach().min().item(), 7), round(v_trace.cpu().detach().mean().item(), 7), round(v_trace.cpu().detach().std().item(), 7), round(v_trace.cpu().detach().max().item(), 7))
            print()

            print("weighted critic loss:", round(float(critic_loss * self.critic_weight), 7))
            print("weighted actor v loss:", round(float(actor_v_loss * self.actor_v_weight), 7))
            print("weighted actor entropy loss:", round(float(actor_entropy_loss * self.actor_entropy_weight), 7))
            print('-----------------------------------------------------------')
