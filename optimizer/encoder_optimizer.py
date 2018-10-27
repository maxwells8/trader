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
from simple_worker import Experience
import networks
from networks import *
from environment import *
import redis
import pickle

class Optimizer(object):

    def __init__(self, models_loc):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        self.models_loc = models_loc

        self.encoder = AttentionMarketEncoder().cuda()
        self.decoder = DecoderProbabilities().cuda()
        try:
            self.encoder.load_state_dict(torch.load(self.models_loc + 'market_encoder.pt'))
            self.decoder.load_state_dict(torch.load(self.models_loc + 'decoder.pt'))
        except FileNotFoundError:
            torch.save(self.encoder.state_dict(), self.models_loc + 'market_encoder.pt')
            torch.save(self.decoder.state_dict(), self.models_loc + 'decoder.pt')

        self.server = redis.Redis("localhost")
        self.weight_penalty = float(self.server.get("weight_penalty").decode("utf-8"))
        self.learning_rate = float(self.server.get("learning_rate").decode("utf-8"))
        self.batch_size = int(self.server.get("queued_batch_size").decode("utf-8"))

        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))
        self.window = networks.WINDOW

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate, weight_decay=self.weight_penalty)

    def run(self):
        n_experiences = 0
        step = 1
        while True:
            # read in experience from the queue
            experiences = []
            while True:
                if (len(experiences) < self.batch_size and self.server.llen("experience") > 0):
                    experience = self.server.lpop("experience")
                    experience = pickle.loads(experience)
                    experiences.append(experience)
                    n_experiences += 1
                elif (step != 1 or (step == 1 and len(experiences) == self.batch_size)) and len(experiences) > 0:
                    break
                else:
                    experience = self.server.blpop("experience")[1]
                    experience = pickle.loads(experience)
                    experiences.append(experience)
                    n_experiences += 1

            self.optimizer.zero_grad()

            batch = Experience(*zip(*experiences))
            time_states = [*zip(*batch.time_states)]
            for i, time_state_ in enumerate(time_states):
                time_states[i] = torch.cat(time_state_)
            spread = [*zip(*batch.spreads)]

            total_loss = torch.Tensor([0])

            for i in range(1, self.trajectory_steps):
                time_states_ = torch.cat(time_states[-self.window-i:-i], dim=1).cuda()
                mean = time_states_[:, :, :4].contiguous().view(len(experiences), -1).mean(1).view(len(experiences), 1, 1)
                std = time_states_[:, :, :4].contiguous().view(len(experiences), -1).std(1).view(len(experiences), 1, 1)
                time_states_[:, :, :4] = (time_states_[:, :, :4] - mean) / std
                spread_ = torch.Tensor(spread[-i]).view(-1, 1, 1).cuda() / std
                time_states_ = time_states_.transpose(0, 1)

                market_encoding = self.encoder.forward(time_states_, spread_)
                probabilities = self.decoder.forward(market_encoding, torch.Tensor([i]).repeat(market_encoding.size()[0], 1).log().cuda())

                log_prob_buy = torch.log(probabilities.gather(1, torch.zeros(probabilities.size()[0], 1).long())).cuda()
                potential_gain_buy = torch.cat(time_states[-i:], dim=1)[:,:,3].cuda().max(1)[0].view(-1, 1)
                potential_gain_buy -= time_states[-i][:,:,3].cuda() - torch.Tensor(spread[-i]).view(-1, 1) / 2
                potential_gain_buy = potential_gain_buy / (std.view(-1, 1) * math.sqrt(len(time_states[-i:])))

                log_prob_sell = torch.log(probabilities.gather(1, torch.ones(probabilities.size()[0], 1).long())).cuda()
                potential_gain_sell = time_states[-i][:,:,3].cuda() + torch.Tensor(spread[-i]).view(-1, 1) / 2
                potential_gain_sell -= torch.cat(time_states[-i:], dim=1)[:,:,3].cuda().min(1)[0].view(-1, 1)
                potential_gain_sell = potential_gain_sell / (std.view(-1, 1) * math.sqrt(len(time_states[-i:])))

                log_prob_stay = torch.log(probabilities.gather(1, 2*torch.ones(probabilities.size()[0], 1).long())).cuda()
                potential_gain_stay = torch.zeros_like(potential_gain_buy)

                actor_pot_mean = (potential_gain_buy + potential_gain_sell + potential_gain_stay) / 3

                advantage_buy = potential_gain_buy - actor_pot_mean
                advantage_sell = potential_gain_sell - actor_pot_mean
                advantage_stay = potential_gain_stay - actor_pot_mean

                actor_pot_loss_buy = -log_prob_buy * advantage_buy.detach()
                actor_pot_loss_sell = -log_prob_sell * advantage_sell.detach()
                actor_pot_loss_stay = -log_prob_stay * advantage_stay.detach()

                total_loss += (actor_pot_loss_buy.mean() + actor_pot_loss_sell.mean() + actor_pot_loss_stay.mean()) / self.trajectory_steps

            total_loss.backward()
            self.optimizer.step()

            print("n experiences: {n}, steps: {s}".format(n=n_experiences, s=step))
            print("loss: {l}".format(l=total_loss))

            try:
                torch.save(self.encoder.state_dict(), self.models_loc + "market_encoder.pt")
                torch.save(self.decoder.state_dict(), self.models_loc + "decoder.pt")
            except Exception:
                print("failed to save")
