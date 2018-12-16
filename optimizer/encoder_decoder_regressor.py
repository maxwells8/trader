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
        self.decoder = Decoder().cuda()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate, weight_decay=self.weight_penalty)
        self.start_n_samples = 0
        self.start_step = 0
        self.start_pos_ema = 0
        self.start_neg_ema = 0
        self.start_value_ema = 0
        try:
            self.encoder.load_state_dict(torch.load(self.models_loc + 'market_encoder.pt'))
            self.decoder.load_state_dict(torch.load(self.models_loc + 'decoder.pt'))
            checkpoint = torch.load(self.models_loc + 'encoder_train.pt')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_step = checkpoint['steps']
            self.start_n_samples = checkpoint['n_samples']
            self.start_pos_ema = checkpoint['pos_ema']
            self.start_neg_ema = checkpoint['neg_ema']
            self.start_value_ema = checkpoint['value_ema']
            self.start_value_ema = 0
        except Exception as e:
            print(e)
            torch.save(self.encoder.state_dict(), self.models_loc + 'market_encoder.pt')
            torch.save(self.decoder.state_dict(), self.models_loc + 'decoder.pt')
            self.start_n_samples = 0
            self.start_step = 0
            self.start_pos_ema = 0
            self.start_neg_ema = 0
            self.start_value_ema = 0
            cur_state = {
                'n_samples':self.start_n_samples,
                'steps':self.start_step,
                'p profit':self.start_pos_ema,
                'p loss':self.start_neg_ema,
                'value_ema':self.start_value_ema,
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

        pos_neg_tau = 0.0001
        pos_ema = self.start_pos_ema
        neg_ema = self.start_neg_ema

        value_tau = 0.0001
        value_ema = self.start_value_ema

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
            final_time_states = torch.Tensor(batch.final_time_state).view(self.batch_size, 1, networks.D_BAR)
            final_spreads = batch.final_spread
            steps = torch.Tensor(batch.steps)

            total_loss = torch.Tensor([0])

            input_time_states_ = torch.cat(input_time_states, dim=1).clone().cuda()
            mean = input_time_states_[:, :, :4].contiguous().view(len(experiences), -1).mean(1).view(len(experiences), 1, 1)
            std = input_time_states_[:, :, :4].contiguous().view(len(experiences), -1).std(1).view(len(experiences), 1, 1)
            input_time_states_[:, :, :4] = (input_time_states_[:, :, :4] - mean) / std
            spread_ = torch.Tensor(initial_spreads).view(-1, 1, 1).cuda() / std
            input_time_states_ = input_time_states_.transpose(0, 1)

            market_encoding = self.encoder.forward(input_time_states_)
            value_estimation = self.decoder.forward(market_encoding, spread_, steps.log().cuda())

            # since the data time_state is using the bid price, we calculate
            # the normalized profit as follows
            future_value = final_time_states[:,:,3].cuda()

            potential_gain_buy = future_value.clone()
            potential_gain_buy -= input_time_states[-1][:,:,3].cuda()
            potential_gain_buy -= torch.Tensor(initial_spreads).view(-1, 1)
            potential_gain_buy = potential_gain_buy / (std.view(-1, 1) * torch.sqrt(steps).view(-1, 1))

            potential_gain_sell = input_time_states[-1][:,:,3].clone().cuda()
            potential_gain_sell -= future_value.clone()
            potential_gain_sell -= torch.Tensor(final_spreads).view(-1, 1)
            potential_gain_sell = potential_gain_sell / (std.view(-1, 1) * torch.sqrt(steps).view(-1, 1))

            actor_pot_loss_buy = F.l1_loss(value_estimation[:, 0].view(-1, 1), potential_gain_buy.detach())
            actor_pot_loss_sell = F.l1_loss(value_estimation[:, 1].view(-1, 1), potential_gain_sell.detach())

            total_loss += actor_pot_loss_buy.mean() + actor_pot_loss_sell.mean()

            # print(value_estimation[:, 0], potential_gain_buy.squeeze())
            # print(value_estimation[:, 1], potential_gain_sell.squeeze())
            # print()
            value = 0
            for j in range(self.batch_size):
                guesses = [float(value_estimation[j, 0]), float(value_estimation[j, 1]), 0]
                targets = [float(potential_gain_buy[j, 0]), float(potential_gain_sell[j, 0]), 0]

                # print(guesses)
                # print(targets)
                # print()

                if np.argmax(guesses) == 0:
                    value = float(potential_gain_buy[j] * (std.view(-1)[j] * torch.sqrt(steps.view(-1)[j])))
                elif np.argmax(guesses) == 1:
                    value = float(potential_gain_sell[j] * (std.view(-1)[j] * torch.sqrt(steps.view(-1)[j])))
                else:
                    value = 0

                value_ema = (value_tau * value) + ((1 - value_tau) * value_ema)

                positive = False
                negative = False
                if value > 0:
                    positive = True
                elif value < 0:
                    negative = True

                pos_ema = positive * pos_neg_tau + pos_ema * (1 - pos_neg_tau)
                neg_ema = negative * pos_neg_tau + neg_ema * (1 - pos_neg_tau)

            if loss_ema == 0:
                loss_ema = float(total_loss)
            else:
                loss_ema = float(total_loss) * loss_tau + (loss_ema) * (1 - loss_tau)

            assert torch.isnan(total_loss).sum() == 0

            total_loss.backward()
            self.optimizer.step()

            step += 1
            n_samples += n_experiences

            print("n samples: {n}, steps: {s}, time ema: {t}, loss ema: {l}, gain ema: {v}, pos ema: {pos}, neg ema: {neg}".format(
                n=n_samples,
                s=step,
                t=round(t, 5),
                l=round(loss_ema, 5),
                v=round(value_ema, 8),
                pos=round(pos_ema, 5),
                neg=round(neg_ema, 5)
            ))

            if step % self.n_steps_to_save == 0:
                try:
                    torch.save(self.encoder.state_dict(), self.models_loc + "market_encoder.pt")
                    torch.save(self.decoder.state_dict(), self.models_loc + "decoder.pt")
                    cur_state = {
                        'n_samples':n_samples,
                        'steps':step,
                        'pos_ema':pos_ema,
                        'neg_ema':neg_ema,
                        'value_ema':value_ema,
                        'optimizer':self.optimizer.state_dict()
                    }
                    torch.save(cur_state, self.models_loc + 'encoder_train.pt')
                except Exception as e:
                    print("failed to save")

            if t == 0:
                t = time.time() - t0
            t = (time.time() - t0) * t_tau + t * (1 - t_tau)
            t0 = time.time()
