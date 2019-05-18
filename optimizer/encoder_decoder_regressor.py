import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import time
import sys
sys.path.insert(0, '../worker')
from simple_worker_regressor import Experience
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

        self.generator = Generator().cuda()
        self.discriminator = Discriminator().cuda()
        self.generator_optimizer = optim.Adam([param for param in self.generator.parameters()],
                                            weight_decay=self.weight_penalty)
        self.discriminator_optimizer = optim.Adam([param for param in self.discriminator.parameters()],
                                            weight_decay=self.weight_penalty)

        try:
            # models
            self.generator.load_state_dict(torch.load(self.models_loc + 'generator.pt'))
            self.discriminator.load_state_dict(torch.load(self.models_loc + 'discriminator.pt'))

            # optimizer
            checkpoint = torch.load(self.models_loc + "meta_state.pt")

            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
            self.start_step = checkpoint['steps']
            self.start_n_samples = checkpoint['n_samples']

            meta_state_buffer = io.BytesIO()
            torch.save(checkpoint, meta_state_buffer)
            torch.save(checkpoint, self.models_loc + 'meta_state.pt')

        except (FileNotFoundError, AssertionError) as e:
            self.generator = Generator().cuda()
            self.discriminator = Discriminator().cuda()
            self.generator_optimizer = optim.Adam([param for param in self.generator.parameters()],
                                                weight_decay=self.weight_penalty)
            self.discriminator_optimizer = optim.Adam([param for param in self.discriminator.parameters()],
                                                weight_decay=self.weight_penalty)

            torch.save(self.generator.state_dict(), self.models_loc + 'generator.pt')
            torch.save(self.discriminator.state_dict(), self.models_loc + 'discriminator.pt')

            self.start_step = 0
            self.start_n_samples = 0
            cur_meta_state = {
                'n_samples':self.start_n_samples,
                'steps':self.start_step,
                'generator_optimizer':self.generator_optimizer.state_dict(),
                'discriminator_optimizer':self.discriminator_optimizer.state_dict()
            }

            meta_state_buffer = io.BytesIO()
            torch.save(cur_meta_state, meta_state_buffer)
            torch.save(cur_meta_state, self.models_loc + 'meta_state.pt')

            cur_meta_state_compressed = pickle.dumps(cur_meta_state)
            self.server.set("optimizer", cur_meta_state_compressed)

        self.loss = nn.BCELoss()

        self.n_samples = self.start_n_samples
        self.step = self.start_step

        self.n_disc_to_gen_steps = 1

        n = 0
        for param in self.generator.parameters():
            n += np.prod(param.size())
        print("generator parameters:", n)
        n = 0
        for param in self.discriminator.parameters():
            n += np.prod(param.size())
        print("discriminator parameters:", n)

        self.batch_size = int(self.server.get("queued_batch_size").decode("utf-8"))

        self.window = networks.WINDOW

    def set_learning_rate(self):
        n_warm_up = 2000
        if self.step < n_warm_up:
            lr = self.step * self.base_learning_rate / n_warm_up
        else:
            lr = self.base_learning_rate
        # lr = self.base_learning_rate
        print("learning rate: {lr}".format(lr=lr))
        for optimizer in [self.generator_optimizer, self.discriminator_optimizer]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def run(self):
        self.generator.train()
        self.discriminator.train()

        while True:
            t0 = time.time()

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
                # if len(experiences) < self.batch_size:
                #     experience = self.server.lindex("experience_dev", len(experiences))
                #     experience = msgpack.unpackb(experience, raw=False)
                #     experiences.append(experience)
                #     n_experiences += 1
                # else:
                #     break

            np.random.shuffle(experiences)
            batch_size = len(experiences)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            batch = Experience(*zip(*experiences))
            time_states = [*zip(*batch.time_states)]
            for i, time_state_ in enumerate(time_states):
                time_states[i] = torch.Tensor(time_state_).view(batch_size, 1, networks.D_BAR)

            input_time_states = torch.cat(time_states[:self.window], dim=1).cuda()

            next_time_states = torch.cat(time_states[self.window:], dim=1).cuda()
            n_future = next_time_states.size()[1]

            generation = self.generator(input_time_states[:(batch_size // 2)])

            queried = torch.cat([generation, next_time_states[(batch_size // 2):, :, 3].view((batch_size // 2), -1, 1)], dim=0)
            discrimination = self.discriminator(input_time_states[:, :, 3].view(batch_size, -1, 1), queried)

            discriminator_loss = self.loss(discrimination, torch.cat([torch.zeros(batch_size // 2, 1), torch.ones(batch_size // 2, 1)], 0))

            if self.step % self.n_disc_to_gen_steps == 0:
                discriminator_loss.backward(retain_graph=True)
            else:
                discriminator_loss.backward(retain_graph=False)
            self.discriminator_optimizer.step()

            self.generator_optimizer.zero_grad()
            generator_loss = self.loss(discrimination[:(batch_size // 2)], torch.ones(batch_size // 2, 1))

            if self.step % self.n_disc_to_gen_steps == 0:
                generator_loss.backward(retain_graph=False)
                self.generator_optimizer.step()

            if self.step % 10 == 0:
                print("generation mean:", generation.view(-1, D_BAR).mean(0).cpu().detach().numpy())
                print("actual mean:", next_time_states.view(-1, D_BAR).mean(0).cpu().detach().numpy())
                print("sample generation:", generation[0][0].cpu().detach().numpy())
                print("sample actual:", next_time_states[0][0][-3].cpu().detach().numpy())
                print("sample discrimination:", discrimination[0][0].cpu().detach().numpy())
            print("step: {step}, n_samples: {n_samples}, time: {time}, dis_target: {dis_target}, dis_gen: {dis_gen}, gen_loss: {gen_loss}, dis_loss: {dis_loss}".format(step=self.step,
                n_samples=self.n_samples,
                time=round(time.time() - t0, 5),
                dis_target=round(discrimination[(batch_size // 2):].mean().item(), 5),
                dis_gen=round(discrimination[:(batch_size // 2)].mean().item(), 5),
                gen_loss=round(generator_loss.cpu().item(), 5),
                dis_loss=round(discriminator_loss.cpu().item(), 5)))
            print('-----------------------------------------------------------------------')

            if self.step % 100 == 0:
                try:
                    cur_meta_state = {
                        'n_samples':self.n_samples,
                        'steps':self.step,
                        'generator_optimizer':self.generator_optimizer.state_dict(),
                        'discriminator_optimizer':self.discriminator_optimizer.state_dict()
                    }

                    torch.save(self.generator.state_dict(), self.models_loc + 'generator.pt')
                    torch.save(self.discriminator.state_dict(), self.models_loc + 'discriminator.pt')
                    torch.save(cur_meta_state, self.models_loc + 'meta_state.pt')
                except Exception:
                    print("failed to save")

            if self.step % 10000 == 0:
                try:
                    if not os.path.exists(self.models_loc + 'model_history'):
                        os.makedirs(self.models_loc + 'model_history')
                    if not os.path.exists(self.models_loc + 'model_history/{step}'.format(step=self.step)):
                        os.makedirs(self.models_loc + 'model_history/{step}'.format(step=self.step))
                    torch.save(self.generator.state_dict(), self.models_loc + 'model_history/{step}/generator.pt'.format(step=self.step))
                    torch.save(self.discriminator.state_dict(), self.models_loc + 'model_history/{step}/discriminator.pt'.format(step=self.step))
                    cur_meta_state = {
                        'n_samples':self.n_samples,
                        'steps':self.step,
                        'generator_optimizer':self.generator_optimizer.state_dict(),
                        'discriminator_optimizer':self.discriminator_optimizer.state_dict()
                    }
                    torch.save(cur_meta_state, self.models_loc + 'model_history/{step}/meta_state.pt'.format(step=self.step))

                except Exception as e:
                    print(e)
                    assert False
                    print("failed to save")


            self.step += 1
            self.n_samples += n_experiences
