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
from simple_worker_classifier import Experience
import networks
from networks import *
from environment import *
import redis
import msgpack

class Optimizer(object):

    def __init__(self, models_loc):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        self.server = redis.Redis("localhost")
        self.weight_penalty = float(self.server.get("weight_penalty").decode("utf-8"))
        self.learning_rate = float(self.server.get("learning_rate").decode("utf-8"))
        self.n_steps_to_save = int(self.server.get("n_steps_to_save").decode("utf-8"))

        self.models_loc = models_loc

        self.encoder = CNNEncoder().cuda()
        self.decoder = ClassifierDecoder().cuda()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate, weight_decay=self.weight_penalty)
        self.start_n_samples = 0
        self.start_step = 0
        try:
            self.encoder.load_state_dict(torch.load(self.models_loc + 'market_encoder.pt'))
            self.decoder.load_state_dict(torch.load(self.models_loc + 'decoder.pt'))
            checkpoint = torch.load(self.models_loc + 'encoder_train.pt')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_step = checkpoint['steps']
            self.start_n_samples = checkpoint['n_samples']
            self.start_value_ema = 0
        except Exception as e:
            print(e)
            torch.save(self.encoder.state_dict(), self.models_loc + 'market_encoder.pt')
            torch.save(self.decoder.state_dict(), self.models_loc + 'decoder.pt')
            self.start_n_samples = 0
            self.start_step = 0
            cur_state = {
                'n_samples':self.start_n_samples,
                'steps':self.start_step,
                'optimizer':self.optimizer.state_dict()
            }
            torch.save(cur_state, self.models_loc + 'encoder_train.pt')

        n_param_encoder = 0
        for param in self.encoder.parameters():
            n_param_ = 1
            for size in param.size():
                n_param_ *= size
            n_param_encoder += n_param_
        print("encoder number of parameters:", n_param_encoder)

        self.advantage_weight = float(self.server.get("advantage_weight").decode("utf-8"))
        self.batch_size = int(self.server.get("queued_batch_size").decode("utf-8"))
        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.window = networks.WINDOW

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate, weight_decay=self.weight_penalty)

    def run(self):
        n_samples = self.start_n_samples
        step = self.start_step
        t0 = time.time()

        loss_ema = 0
        loss_tau = 0.01

        t = 0
        t_tau = 0.05

        p = 0.5
        p_tau = 0.01

        p_bull = 0
        p_bear = 0
        p_change_tau = 0.01

        while True:
            n_experiences = 0
            # read in experience from the queue
            experiences = []
            t_tot = 0
            while True:
                if len(experiences) < self.batch_size:
                    experience = self.server.blpop("experience")[1]
                    experience = msgpack.unpackb(experience, raw=False)
                    experiences.append(experience)
                    n_experiences += 1
                else:
                    break

            self.optimizer.zero_grad()

            batch = Experience(*zip(*experiences))
            input_time_states = [*zip(*batch.input_time_states)]
            for i, time_state_ in enumerate(input_time_states):
                input_time_states[i] = torch.Tensor(time_state_).view(self.batch_size, 1, networks.D_BAR)
            initial_spreads = batch.initial_spread
            confidence_interval = torch.Tensor(batch.confidence_interval)
            targets = torch.Tensor(batch.output).long()
            # print(targets)

            total_loss = torch.Tensor([0])

            input_time_states_ = torch.cat(input_time_states, dim=1).clone().cuda()
            mean = input_time_states_[:, :, :4].contiguous().view(len(experiences), -1).mean(1).view(len(experiences), 1, 1)
            std = input_time_states_[:, :, :4].contiguous().view(len(experiences), -1).std(1).view(len(experiences), 1, 1)
            input_time_states_[:, :, :4] = (input_time_states_[:, :, :4] - mean) / std
            spread_ = torch.Tensor(initial_spreads).view(-1, 1, 1).cuda() / std
            input_time_states_ = input_time_states_.transpose(0, 1)

            market_encoding = self.encoder.forward(input_time_states_)
            probability_estimation = self.decoder.forward(market_encoding, spread_ * std, std, confidence_interval)
            print(probability_estimation, targets)

            total_loss += F.cross_entropy(probability_estimation, targets)

            if loss_ema == 0:
                loss_ema = float(total_loss)
            else:
                loss_ema = float(total_loss) * loss_tau + (loss_ema) * (1 - loss_tau)

            assert torch.isnan(total_loss).sum() == 0

            total_loss.backward()
            self.optimizer.step()

            step += 1
            n_samples += n_experiences

            self.n_correct = 0
            for i in range(self.batch_size):
                # print(probability_estimation[i], targets[i])
                if torch.argmax(probability_estimation[i]) == targets[i]:
                    self.n_correct += 1

            p = (self.n_correct / self.batch_size) * p_tau + p * (1 - p_tau)
            p_bull = (np.array(targets) == 0).sum() / self.batch_size * p_change_tau + p_bull * (1 - p_change_tau)
            p_bear = (np.array(targets) == 1).sum() / self.batch_size * p_change_tau + p_bear * (1 - p_change_tau)

            print("n samples: {n}, steps: {s}, time ema: {t}, loss ema: {l}, p correct: {p}, p bull: {p_bull}, p bear: {p_bear}".format(
                n=n_samples,
                s=step,
                t=round(t, 5),
                l=round(loss_ema, 5),
                p=round(p, 5),
                p_bull=round(p_bull, 5),
                p_bear=round(p_bear, 5),
            ))


            if step % self.n_steps_to_save == 0:
                try:
                    torch.save(self.encoder.state_dict(), self.models_loc + "market_encoder.pt")
                    torch.save(self.decoder.state_dict(), self.models_loc + "decoder.pt")
                    cur_state = {
                        'n_samples':n_samples,
                        'steps':step,
                        'optimizer':self.optimizer.state_dict()
                    }
                    torch.save(cur_state, self.models_loc + 'encoder_train.pt')
                except Exception as e:
                    print("failed to save")

            if t == 0:
                t = time.time() - t0
            t = (time.time() - t0) * t_tau + t * (1 - t_tau)
            t0 = time.time()
