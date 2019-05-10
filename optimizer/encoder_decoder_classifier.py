import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import time
import sys
sys.path.insert(0, '../worker')
from simple_worker_classifier import Experience
from networks import *
import networks
import io
import pickle
import redis
import msgpack
import os

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Optimizer(object):

    def __init__(self, models_loc, server_host):

        self.server = redis.Redis(server_host)
        self.models_loc = models_loc
        self.base_learning_rate = float(self.server.get("learning_rate").decode("utf-8"))
        self.weight_penalty = float(self.server.get("weight_penalty").decode("utf-8"))

        self.network = AttentionMarketEncoder().cuda()
        self.optimizer = optim.Adam([param for param in self.network.parameters()],
                                    weight_decay=self.weight_penalty)

        try:
            # models
            self.network.load_state_dict(torch.load(self.models_loc + 'network.pt'))

            # optimizer
            checkpoint = torch.load(self.models_loc + "meta_state.pt")

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_step = checkpoint['steps']
            self.start_n_samples = checkpoint['n_samples']

            meta_state_buffer = io.BytesIO()
            torch.save(checkpoint, meta_state_buffer)
            torch.save(checkpoint, self.models_loc + 'meta_state.pt')

        except (FileNotFoundError, AssertionError) as e:
            self.network = AttentionMarketEncoder().cuda()
            self.optimizer = optim.Adam([param for param in self.network.parameters()],
                                        weight_decay=self.weight_penalty)

            torch.save(self.network.state_dict(), self.models_loc + 'network.pt')

            self.start_step = 0
            self.start_n_samples = 0
            cur_meta_state = {
                'n_samples':self.start_n_samples,
                'steps':self.start_step,
                'optimizer':self.optimizer.state_dict()
            }

            meta_state_buffer = io.BytesIO()
            torch.save(cur_meta_state, meta_state_buffer)
            torch.save(cur_meta_state, self.models_loc + 'meta_state.pt')

            cur_meta_state_compressed = pickle.dumps(cur_meta_state)
            self.server.set("optimizer", cur_meta_state_compressed)

        self.loss = nn.CrossEntropyLoss()

        self.n_samples = self.start_n_samples
        self.step = self.start_step

        n = 0
        for param in self.network.parameters():
            n += np.prod(param.size())
        print("network parameters:", n)

        self.batch_size = int(self.server.get("batch_size").decode("utf-8"))
        self.KL_coef = float(self.server.get("KL_coef").decode("utf-8"))

        self.window = networks.WINDOW
        self.required_change = float(self.server.get("required_change").decode("utf-8"))

        self.acc_ema = 0.9
        self.acc_tau = 0.01

    def set_learning_rate(self):
        n_warmup_steps = 2000
        if self.step < n_warmup_steps:
            lr = self.step * self.base_learning_rate / n_warmup_steps
        else:
            lr = self.base_learning_rate
        # lr = self.base_learning_rate
        # print("learning rate: {lr}".format(lr=lr))
        for optimizer in [self.optimizer, self.optimizer]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def run(self):
        self.network.train()

        dev_experiences = []
        n_dev_exp = self.server.llen("experience_dev")
        n_dev_exp = 10000
        while len(dev_experiences) < n_dev_exp:
            experience = self.server.lindex("experience_dev", len(dev_experiences))
            experience = msgpack.unpackb(experience, raw=False)
            dev_experiences.append(experience)
        np.random.shuffle(dev_experiences)

        dev_batch_size = n_dev_exp
        dev_batch = Experience(*zip(*dev_experiences))
        dev_time_states = [*zip(*dev_batch.time_states)]
        dev_input_time_states = []
        for i in range(self.window):
            dev_input_time_states.append(torch.Tensor(dev_time_states[i]).view(dev_batch_size, 1, networks.D_BAR))
        dev_final_time_state = torch.Tensor(dev_time_states[-1]).view(dev_batch_size, networks.D_BAR)

        dev_input_time_states_ = torch.cat(dev_input_time_states, dim=1).clone().cuda()
        mean = dev_input_time_states_[:, :, :4].contiguous().view(dev_batch_size, -1).mean(1).view(dev_batch_size, 1, 1)
        dev_input_time_states_[:, :, :4] = (dev_input_time_states_[:, :, :4] - mean) * 1000 / mean
        dev_input_time_states_ = dev_input_time_states_.transpose(0, 1)

        mean = mean.view(dev_batch_size, 1)
        dev_final_time_state_ = (dev_final_time_state.cuda() - mean) * 1000 / mean
        dev_actual = (dev_final_time_state_[:, 3] > dev_input_time_states_[-1, :, 3]).long()

        experience_i = 0
        epoch = 0
        while True:
            t0 = time.time()

            self.set_learning_rate()

            n_experiences = 0
            # read in experience from the queue
            experiences = []

            # while True:
            #     if len(experiences) < self.batch_size:
            #         experience = self.server.blpop("experience")[1]
            #         experience = msgpack.unpackb(experience, raw=False)
            #         experiences.append(experience)
            #         n_experiences += 1
            #     else:
            #         break
            while len(experiences) < self.batch_size:
                experiences.append(dev_experiences[experience_i])
                experience_i += 1
                if experience_i >= dev_batch_size:
                    epoch += 1
                    experience_i %= dev_batch_size
                    np.random.shuffle(dev_experiences)

            batch_size = len(experiences)

            batch = Experience(*zip(*experiences))
            time_states = [*zip(*batch.time_states)]
            input_time_states = []
            for i in range(self.window):
                input_time_states.append(torch.Tensor(time_states[i]).view(batch_size, 1, networks.D_BAR))
            final_time_state = torch.Tensor(time_states[-1]).view(batch_size, networks.D_BAR)

            self.optimizer.zero_grad()

            total_loss = torch.Tensor([0])
            loss = torch.Tensor([0])
            kl_loss = torch.Tensor([0])

            constant_scale = 1000
            input_time_states_ = torch.cat(input_time_states, dim=1).clone().cuda()
            mean = input_time_states_[:, :, :4].contiguous().view(batch_size, -1).mean(1).view(batch_size, 1, 1)
            input_time_states_[:, :, :4] = (input_time_states_[:, :, :4] - mean) * constant_scale / mean
            input_time_states_ = input_time_states_.transpose(0, 1)

            mean = mean.view(batch_size, 1)
            final_time_state_ = (final_time_state.cuda() - mean) * constant_scale / mean

            actual = torch.ones(batch_size).long()
            short_flag = ((final_time_state_[:, 3] - input_time_states_[-1, :, 3]) <= (-self.required_change * constant_scale)).long()
            long_flag = ((final_time_state_[:, 3] - input_time_states_[-1, :, 3]) >= (self.required_change * constant_scale)).long()
            # actual = (final_time_state_[:, 3] > input_time_states_[-1, :, 3]).long()
            actual += -1 * short_flag + 1 * long_flag
            actual = actual.long()
            # print((actual == 0).float().mean())
            # print((actual == 1).float().mean())
            # print((actual == 2).float().mean())

            prediction = self.network(input_time_states_)
            # print((prediction.argmax(1) == 0).float().mean())
            # print((prediction.argmax(1) == 1).float().mean())
            # print((prediction.argmax(1) == 2).float().mean())
            # print(prediction.mean(0))

            prediction_loss = self.loss(prediction, actual)

            prediction_loss.backward(retain_graph=False)
            self.optimizer.step()

            self.step += 1
            self.n_samples += n_experiences

            # if self.step % 100 == 0:
            #     accuracy = 0
            #     n_mini_batches = 8
            #     predicted_means = np.zeros(2)
            #     for i_mini_batch in range(n_mini_batches):
            #         start = i_mini_batch * dev_batch_size // n_mini_batches
            #         end = (i_mini_batch + 1) * dev_batch_size // n_mini_batches
            #         predicted = self.network(dev_input_time_states_[:, start:end])
            #         predicted_means += prediction.mean(0).detach().cpu().numpy() / n_mini_batches
            #         accuracy += (predicted.argmax(1) == dev_actual[start:end]).float().mean().detach().item()
            #     accuracy /= n_mini_batches
            #
            #     print("accuracy:", accuracy)
            #     print(predicted_means)

            cur_acc = (prediction.argmax(1) == actual).float().mean().detach().item()
            self.acc_ema = self.acc_tau * cur_acc + (1 - self.acc_tau) * self.acc_ema

            print("step: {step}, epoch: {epoch}, time: {time}, loss: {loss}, running acc: {acc}".format(
                step=self.step,
                epoch=epoch,
                time=round(time.time() - t0, 5),
                loss=round(prediction_loss.cpu().item(), 5),
                acc=round(self.acc_ema, 5)
            ))
            print('-------------------------------------------------------')

            if self.step % 100 == 0:
                try:
                    cur_meta_state = {
                        'n_samples':self.n_samples,
                        'steps':self.step,
                        'optimizer':self.optimizer.state_dict()
                    }

                    torch.save(self.network.state_dict(), self.models_loc + 'network.pt')
                    torch.save(cur_meta_state, self.models_loc + 'meta_state.pt')
                except Exception:
                    print("failed to save")

            if self.step % 10000 == 0:
                try:
                    if not os.path.exists(self.models_loc + 'model_history'):
                        os.makedirs(self.models_loc + 'model_history')
                    if not os.path.exists(self.models_loc + 'model_history/{step}'.format(step=self.step)):
                        os.makedirs(self.models_loc + 'model_history/{step}'.format(step=self.step))
                    torch.save(self.network.state_dict(), self.models_loc + 'model_history/{step}/network.pt'.format(step=self.step))
                    cur_meta_state = {
                        'n_samples':self.n_samples,
                        'steps':self.step,
                        'optimizer':self.optimizer.state_dict()
                    }
                    torch.save(cur_meta_state, self.models_loc + 'model_history/{step}/meta_state.pt'.format(step=self.step))

                except Exception as e:
                    print(e)
                    assert False
                    print("failed to save")
