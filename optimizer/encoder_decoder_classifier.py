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

        self.network = Network().cuda()
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
            self.network = Network().cuda()
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

        self.acc_ema = None
        self.acc_tau = 0.01

        self.best_dev_acc = 0
        self.best_dev_acc_step = 0

    def set_learning_rate(self):
        n_warmup_steps = 0
        if self.step < n_warmup_steps:
            lr = self.step * self.base_learning_rate / n_warmup_steps
        else:
            lr = self.base_learning_rate
        # lr = self.base_learning_rate

        for optimizer in [self.optimizer, self.optimizer]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print("learning rate: {lr}".format(lr=lr))

    def run(self):
        self.network.train()

        dev_experiences = []
        n_dev_exp = self.server.llen("experience_dev")
        while len(dev_experiences) < n_dev_exp:
            experience = self.server.lindex("experience_dev", len(dev_experiences))
            experience = msgpack.unpackb(experience, raw=False)
            dev_experiences.append(experience)
        np.random.shuffle(dev_experiences)

        dev_batch_size = n_dev_exp
        dev_batch = Experience(*zip(*dev_experiences))
        dev_time_states = [*zip(*dev_batch.time_states)]

        dev_input_time_states = torch.Tensor(dev_time_states[:self.window]).transpose(0, 1).contiguous()
        dev_final_time_state = torch.Tensor(dev_time_states[-1]).contiguous()

        dev_mean = dev_input_time_states[:, :, 3].mean(1)
        dev_std = dev_input_time_states[:, :, 3].std(1)

        dev_last_input = (dev_input_time_states[:, -1, 3] - dev_mean) / (dev_std + 1e-9)
        dev_final = (dev_final_time_state[:, 3] - dev_mean) / (dev_std + 1e-9)

        dev_actual = torch.ones(dev_batch_size).long()
        dev_long_flag = (dev_final - dev_last_input > 0.25).long()
        dev_short_flag = (dev_final - dev_last_input < -0.25).long()
        dev_actual += -1 * dev_long_flag + 1 * dev_short_flag
        dev_actual = dev_actual.long()

        experience_i = 0
        epoch = 0
        while True:
            t0 = time.time()

            self.optimizer.zero_grad()
            self.set_learning_rate()

            n_experiences = 0
            # read in experience from the queue
            experiences = []

            while True:
                if len(experiences) < self.batch_size:
                    experience = self.server.blpop("experience")[1]
                    experience = msgpack.unpackb(experience, raw=False)
                    experiences.append(experience)
                    n_experiences += 1
                else:
                    break
            # while len(experiences) < self.batch_size:
            #     experiences.append(dev_experiences[experience_i])
            #     experience_i += 1
            #     if experience_i >= dev_batch_size:
            #         epoch += 1
            #         experience_i %= dev_batch_size
            #         np.random.shuffle(dev_experiences)

            batch_size = len(experiences)

            batch = Experience(*zip(*experiences))
            time_states = [*zip(*batch.time_states)]

            input_time_states = torch.Tensor(time_states[:self.window]).transpose(0, 1).contiguous()
            final_time_state = torch.Tensor(time_states[-1]).contiguous()

            prediction, last_input, mean, std = self.network(input_time_states)
            final = (final_time_state[:, 3].view(batch_size, 1) - mean) / (std + 1e-9)
            actual = torch.ones(batch_size, 1).long()
            long_flag = (final - last_input > 0.25).long()
            short_flag = (final - last_input < -0.25).long()
            actual += -1 * long_flag + 1 * short_flag
            actual = actual[:, 0].long()

            # print(prediction)

            # print((actual == 1).sum().item() / batch_size)

            prediction_loss = self.loss(prediction, actual)

            prediction_loss.backward(retain_graph=False)
            self.optimizer.step()

            self.n_samples += n_experiences

            if self.step % 100 == 0:
                accuracy = 0
                n_mini_batches = 8
                for i_mini_batch in range(n_mini_batches):
                    start = i_mini_batch * dev_batch_size // n_mini_batches
                    end = (i_mini_batch + 1) * dev_batch_size // n_mini_batches
                    predicted, _, _, _ = self.network(dev_input_time_states[start:end])
                    accuracy += (predicted.argmax(1) == dev_actual[start:end]).float().mean().detach().item()
                accuracy /= n_mini_batches

                if accuracy > self.best_dev_acc:
                    self.best_dev_acc = accuracy
                    self.best_dev_acc_step = self.step

                print('******************************************************************')
                print('------------------------------------------------------------------')
                print("current accuracy: {cur}, best accuracy: {best}, best step: {step}".format(
                        cur=round(accuracy, 5),
                        best=round(self.best_dev_acc, 5),
                        step=self.best_dev_acc_step
                ))
                print('------------------------------------------------------------------')
                print('******************************************************************')

            cur_acc = (prediction.argmax(1) == actual).float().mean().detach().item()
            if self.acc_ema is None:
                self.acc_ema = cur_acc
            else:
                self.acc_ema = self.acc_tau * cur_acc + (1 - self.acc_tau) * self.acc_ema

            print("step: {step}, time: {time}, loss: {loss}, running acc: {run_acc}, batch accuracy: {acc}".format(
                step=self.step,
                time=round(time.time() - t0, 5),
                loss=round(prediction_loss.cpu().item(), 5),
                run_acc=round(self.acc_ema, 5),
                acc=round(cur_acc, 5)
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

            self.step += 1
