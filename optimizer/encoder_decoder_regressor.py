import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import time
import sys
sys.path.insert(0, '../worker')
from collections import namedtuple
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

GenExperience = namedtuple('GenExperience', ('experience', 'generated_states'))

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
            start_step = checkpoint['steps']
            start_n_samples = checkpoint['n_samples']
            start_disc_steps = checkpoint['disc_steps']
            start_gen_steps = checkpoint['gen_steps']

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

            start_step = 0
            start_n_samples = 0
            start_disc_steps = 0
            start_gen_steps = 0
            cur_meta_state = {
                'n_samples':start_n_samples,
                'steps':start_step,
                'disc_steps':start_disc_steps,
                'gen_steps':start_gen_steps,
                'generator_optimizer':self.generator_optimizer.state_dict(),
                'discriminator_optimizer':self.discriminator_optimizer.state_dict()
            }

            meta_state_buffer = io.BytesIO()
            torch.save(cur_meta_state, meta_state_buffer)
            torch.save(cur_meta_state, self.models_loc + 'meta_state.pt')

            cur_meta_state_compressed = pickle.dumps(cur_meta_state)
            self.server.set("optimizer", cur_meta_state_compressed)

        self.loss = nn.BCELoss()

        self.n_samples = start_n_samples
        self.step = start_step
        self.disc_step = start_disc_steps
        self.gen_step = start_gen_steps

        self.disc_diff_ema = 0
        self.disc_diff_tau = 0.1

        n = 0
        for param in self.generator.parameters():
            n += np.prod(param.size())
        print("generator parameters:", n)
        n = 0
        for param in self.discriminator.parameters():
            n += np.prod(param.size())
        print("discriminator parameters:", n)

        self.batch_size = int(self.server.get("queued_batch_size").decode("utf-8"))
        self.gen_replay_batch_size = self.batch_size // 4

        self.window = networks.WINDOW

    def add_gen_experience(self, experience, generated_states):
        gen_exp = GenExperience(experience=experience, generated_states=generated_states)
        gen_exp = msgpack.packb(gen_exp, use_bin_type=True)
        self.server.lpush("gen_experience", gen_exp)
        replay_size = self.server.llen("gen_experience")
        if replay_size > 10000:
            loc = np.random.randint(0, replay_size)
            ref = self.server.lindex("gen_experience", loc)
            self.server.lrem("gen_experience", -1, ref)

    def set_learning_rate(self, disc, gen):
        n_warm_up = 0
        decrease_start_step = 1000000

        if disc:
            if self.disc_step < n_warm_up:
                disc_lr = self.disc_step * self.base_learning_rate / n_warm_up
            elif self.disc_step < decrease_start_step:
                disc_lr = self.base_learning_rate
            else:
                disc_lr = self.base_learning_rate * (0.9999 ** (self.disc_step - decrease_start_step))
        else:
            disc_lr = 0

        if gen:
            if self.gen_step < n_warm_up:
                gen_lr = self.gen_step * self.base_learning_rate / n_warm_up
            elif self.gen_step < decrease_start_step:
                gen_lr = self.base_learning_rate
            else:
                gen_lr = self.base_learning_rate * (0.9999 ** (self.gen_step - decrease_start_step))
        else:
            gen_lr = 0

        for param_group in self.discriminator_optimizer.param_groups:
            param_group['lr'] = disc_lr

        for param_group in self.generator_optimizer.param_groups:
            param_group['lr'] = gen_lr

        print("disc lr: {disc_lr}, gen lr: {gen_lr}".format(disc_lr=round(disc_lr, 12), gen_lr=round(gen_lr, 12)))

    def run(self):
        self.generator.train()
        self.discriminator.train()

        while True:
            t0 = time.time()

            if self.step % 100 == 0:
                try:
                    cur_meta_state = {
                        'n_samples':self.n_samples,
                        'steps':self.step,
                        'disc_steps':self.disc_step,
                        'gen_steps':self.gen_step,
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
                        'disc_steps':self.disc_step,
                        'gen_steps':self.gen_step,
                        'generator_optimizer':self.generator_optimizer.state_dict(),
                        'discriminator_optimizer':self.discriminator_optimizer.state_dict()
                    }
                    torch.save(cur_meta_state, self.models_loc + 'model_history/{step}/meta_state.pt'.format(step=self.step))

                except Exception as e:
                    print(e)
                    assert False
                    print("failed to save")

            n_experiences = 0
            # read in experience from the queue
            experiences = []
            while True:
                if len(experiences) < self.batch_size - self.gen_replay_batch_size:
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

            generated_replays = []
            gen_exp_locs = np.arange(0, self.server.llen("gen_experience"))
            np.random.shuffle(gen_exp_locs)
            gen_exp_locs = gen_exp_locs[:self.gen_replay_batch_size].tolist()
            for gen_exp_loc in gen_exp_locs:
                gen_experience = self.server.lindex("gen_experience", gen_exp_loc)
                gen_experience = msgpack.unpackb(gen_experience, raw=False)
                experiences.append(gen_experience[0])
                generated_replays.append(gen_experience[1])

            generated_replays = torch.Tensor(generated_replays)

            batch_size = len(experiences)
            n_replay_gen = len(gen_exp_locs)
            n_actual = batch_size // 2
            n_new_gen = batch_size - n_replay_gen - n_actual

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            batch = Experience(*zip(*experiences))
            time_states = [*zip(*batch.time_states)]

            input_time_states = torch.Tensor(time_states[:self.window]).transpose(0, 1).contiguous()
            next_time_states = torch.Tensor(time_states[self.window:]).transpose(0, 1).contiguous()
            n_future = next_time_states.size()[1]

            generation = self.generator(input_time_states[:n_new_gen])
            actual = next_time_states[n_new_gen:n_new_gen + n_actual, :, 3].view(n_actual, -1, 1)
            # actual = next_time_states[(batch_size // 2):, -1, 3].view((batch_size // 2), -1, 1)
            # actual = input_time_states[(batch_size // 2):, 10, 3].view((batch_size // 2), 1, 1).repeat(1, 10, 1)

            queried = torch.cat([generation, actual, generated_replays], dim=0)

            discrimination = self.discriminator(input_time_states[:, :, 3].view(batch_size, -1, 1), queried)

            discriminator_loss = self.loss(discrimination, torch.cat([torch.zeros(n_new_gen, 1), torch.ones(n_actual, 1), torch.zeros(n_replay_gen, 1)], 0))
            try:
                assert not torch.isnan(discriminator_loss)
            except AssertionError:
                print(torch.isnan(input_time_states).sum())
                print(torch.isnan(generation).sum())
                print(torch.isnan(discrimination).sum())
                for name, param in self.generator.named_parameters():
                    print(name, torch.isnan(param).sum())
                for name, param in self.discriminator.named_parameters():
                    print(name, torch.isnan(param).sum())
                assert False

            dis_real_mean = discrimination[n_new_gen:n_new_gen + n_actual].mean()
            dis_gen_mean = discrimination[:n_new_gen].mean()
            disc_diff = dis_real_mean - dis_gen_mean

            self.disc_diff_ema = self.disc_diff_tau * disc_diff.item() + (1 - self.disc_diff_tau) * self.disc_diff_ema
            print(self.disc_diff_ema)
            gen_step = self.disc_diff_ema > 0.02
            disc_step = self.disc_diff_ema <= 0.02
            self.set_learning_rate(disc_step, gen_step)

            if disc_step:
                print("disc step")
                self.disc_step += 1
            discriminator_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            self.generator_optimizer.zero_grad()
            generator_loss = self.loss(discrimination[:n_new_gen], torch.ones(n_new_gen, 1))
            try:
                assert not torch.isnan(generator_loss)
            except AssertionError:
                print(torch.isnan(input_time_states).sum())
                print(torch.isnan(generation).sum())
                print(torch.isnan(discrimination).sum())
                for name, param in self.generator.named_parameters():
                    print(name, torch.isnan(param).sum())
                for name, param in self.discriminator.named_parameters():
                    print(name, torch.isnan(param).sum())
                assert False

            if gen_step:
                print("gen step")
                self.gen_step += 1
            generator_loss.backward(retain_graph=False)
            self.generator_optimizer.step()

            for name, param in self.generator.named_parameters():
                try:
                    assert torch.isnan(param).sum() == 0
                except AssertionError:
                    print(name, torch.isnan(param).sum())
                    assert False
            for name, param in self.discriminator.named_parameters():
                try:
                    assert torch.isnan(param).sum() == 0
                except AssertionError:
                    print(name, torch.isnan(param).sum())
                    assert False

            for i in range(n_new_gen):
                self.add_gen_experience(experiences[i], generation[i].detach().tolist())

            # mean = input_time_states[:(batch_size // 2), :, 3].mean(1)
            # std = input_time_states[:(batch_size // 2), :, 3].std(1)
            # gen_normalized = (generation[:, :, 0] - mean.view(batch_size // 2, 1).repeat(1, 10)) / std.view(batch_size // 2, 1).repeat(1, 10)
            # next_normalized = (next_time_states[:(batch_size // 2), :, 3] - mean.view(batch_size // 2, 1).repeat(1, 10)) / std.view(batch_size // 2, 1).repeat(1, 10)
            # my_loss = (gen_normalized - next_normalized).pow(2).mean(1).pow(1/2)

            print("step: {step}, disc steps: {disc_step}, gen steps: {gen_step}, n_samples: {n_samples}, time: {time}, dis_target: {dis_target}, dis_new_gen: {dis_new_gen}, dis_replay_gen: {dis_replay_gen}".format(
                step=self.step,
                disc_step=self.disc_step,
                gen_step=self.gen_step,
                n_samples=self.n_samples,
                time=round(time.time() - t0, 5),
                dis_target=round(discrimination[n_new_gen:n_new_gen + n_actual].mean().item(), 5),
                dis_new_gen=round(discrimination[:n_new_gen].mean().item(), 5),
                dis_replay_gen=round(discrimination[n_new_gen + n_actual:].mean().item(), 5)
                ))
            print('-----------------------------------------------------------------------')

            self.step += 1
            self.n_samples += n_experiences
