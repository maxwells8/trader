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
        self.cond_discriminator = ConditionedDiscriminator().cuda()
        self.uncond_discriminator = UnconditionedDiscriminator().cuda()
        self.generator_optimizer = optim.Adam([param for param in self.generator.parameters()],
                                            weight_decay=self.weight_penalty)
        self.cond_discriminator_optimizer = optim.Adam([param for param in self.cond_discriminator.parameters()],
                                            weight_decay=self.weight_penalty)
        self.uncond_discriminator_optimizer = optim.Adam([param for param in self.uncond_discriminator.parameters()],
                                            weight_decay=self.weight_penalty)

        try:
            # models
            self.generator.load_state_dict(torch.load(self.models_loc + 'generator.pt'))
            self.cond_discriminator.load_state_dict(torch.load(self.models_loc + 'cond_discriminator.pt'))
            self.uncond_discriminator.load_state_dict(torch.load(self.models_loc + 'uncond_discriminator.pt'))

            # optimizer
            checkpoint = torch.load(self.models_loc + "meta_state.pt")

            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
            self.cond_discriminator_optimizer.load_state_dict(checkpoint['cond_discriminator_optimizer'])
            start_step = checkpoint['steps']
            start_n_samples = checkpoint['n_samples']
            start_cond_disc_steps = checkpoint['cond_disc_steps']
            start_uncond_disc_steps = checkpoint['uncond_disc_steps']
            start_gen_steps = checkpoint['gen_steps']

            meta_state_buffer = io.BytesIO()
            torch.save(checkpoint, meta_state_buffer)
            torch.save(checkpoint, self.models_loc + 'meta_state.pt')

        except (FileNotFoundError, AssertionError) as e:
            self.generator = Generator().cuda()
            self.cond_discriminator = ConditionedDiscriminator().cuda()
            self.uncond_discriminator = UnconditionedDiscriminator().cuda()
            self.generator_optimizer = optim.Adam([param for param in self.generator.parameters()],
                                                weight_decay=self.weight_penalty)
            self.cond_discriminator_optimizer = optim.Adam([param for param in self.cond_discriminator.parameters()],
                                                weight_decay=self.weight_penalty)
            self.uncond_discriminator_optimizer = optim.Adam([param for param in self.uncond_discriminator.parameters()],
                                                weight_decay=self.weight_penalty)

            torch.save(self.generator.state_dict(), self.models_loc + 'generator.pt')
            torch.save(self.cond_discriminator.state_dict(), self.models_loc + 'cond_discriminator.pt')
            torch.save(self.uncond_discriminator.state_dict(), self.models_loc + 'uncond_discriminator.pt')

            start_step = 0
            start_n_samples = 0
            start_cond_disc_steps = 0
            start_uncond_disc_steps = 0
            start_gen_steps = 0
            cur_meta_state = {
                'n_samples':start_n_samples,
                'steps':start_step,
                'cond_disc_steps':start_cond_disc_steps,
                'uncond_disc_steps':start_uncond_disc_steps,
                'gen_steps':start_gen_steps,
                'generator_optimizer':self.generator_optimizer.state_dict(),
                'cond_discriminator_optimizer':self.cond_discriminator_optimizer.state_dict(),
                'uncond_discriminator_optimizer':self.uncond_discriminator_optimizer.state_dict()
            }

            meta_state_buffer = io.BytesIO()
            torch.save(cur_meta_state, meta_state_buffer)
            torch.save(cur_meta_state, self.models_loc + 'meta_state.pt')

            cur_meta_state_compressed = pickle.dumps(cur_meta_state)
            self.server.set("optimizer", cur_meta_state_compressed)

        self.loss = nn.BCELoss()

        self.n_samples = start_n_samples
        self.step = start_step
        self.cond_disc_step = start_cond_disc_steps
        self.uncond_disc_step = start_uncond_disc_steps
        self.gen_step = start_gen_steps

        self.cond_disc_diff_ema = 0
        self.uncond_disc_diff_ema = 0
        self.disc_diff_tau = 0.1

        n = 0
        for param in self.generator.parameters():
            n += np.prod(param.size())
        print("generator parameters:", n)
        n = 0
        for param in self.cond_discriminator.parameters():
            n += np.prod(param.size())
        print("cond_discriminator parameters:", n)
        n = 0
        for param in self.uncond_discriminator.parameters():
            n += np.prod(param.size())
        print("uncond_discriminator parameters:", n)

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

    def set_learning_rate(self, cond, uncond, gen):
        n_warm_up = 0
        decrease_start_step = 1000000

        if cond:
            if self.cond_disc_step < n_warm_up:
                cond_disc_lr = self.cond_disc_step * self.base_learning_rate / n_warm_up
            elif self.cond_disc_step < decrease_start_step:
                cond_disc_lr = self.base_learning_rate
            else:
                cond_disc_lr = self.base_learning_rate * (0.9999 ** (self.cond_disc_step - decrease_start_step))
        else:
            cond_disc_lr = 0

        if uncond:
            if self.uncond_disc_step < n_warm_up:
                uncond_disc_lr = self.uncond_disc_step * self.base_learning_rate / n_warm_up
            elif self.uncond_disc_step < decrease_start_step:
                uncond_disc_lr = self.base_learning_rate
            else:
                uncond_disc_lr = self.base_learning_rate * (0.9999 ** (self.uncond_disc_step - decrease_start_step))
        else:
            uncond_disc_lr = 0

        if gen:
            if self.gen_step < n_warm_up:
                gen_lr = self.gen_step * self.base_learning_rate / n_warm_up
            elif self.gen_step < decrease_start_step:
                gen_lr = self.base_learning_rate
            else:
                gen_lr = self.base_learning_rate * (0.9999 ** (self.gen_step - decrease_start_step))
        else:
            gen_lr = 0

        for param_group in self.generator_optimizer.param_groups:
            param_group['lr'] = gen_lr

        for param_group in self.cond_discriminator_optimizer.param_groups:
            param_group['lr'] = cond_disc_lr

        for param_group in self.uncond_discriminator_optimizer.param_groups:
            param_group['lr'] = uncond_disc_lr

        # print("cond disc lr: {c_disc_lr}, uncond disc lr: {u_disc_lr}, gen lr: {gen_lr}".format(c_disc_lr=round(cond_disc_lr, 12), uncond_disc_lr=round(u_disc_lr, 12), gen_lr=round(gen_lr, 12)))

    def run(self):
        self.generator.train()
        self.cond_discriminator.train()
        self.uncond_discriminator.train()

        while True:
            t0 = time.time()

            if self.step % 100 == 0:
                try:
                    cur_meta_state = {
                        'n_samples':self.n_samples,
                        'steps':self.step,
                        'cond_disc_steps':self.cond_disc_step,
                        'uncond_disc_steps':self.uncond_disc_step,
                        'gen_steps':self.gen_step,
                        'generator_optimizer':self.generator_optimizer.state_dict(),
                        'cond_discriminator_optimizer':self.cond_discriminator_optimizer.state_dict(),
                        'uncond_discriminator_optimizer':self.uncond_discriminator_optimizer.state_dict()
                    }

                    torch.save(self.generator.state_dict(), self.models_loc + 'generator.pt')
                    torch.save(self.cond_discriminator.state_dict(), self.models_loc + 'cond_discriminator.pt')
                    torch.save(self.uncond_discriminator.state_dict(), self.models_loc + 'uncond_discriminator.pt')
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
                    torch.save(self.cond_discriminator.state_dict(), self.models_loc + 'model_history/{step}/cond_discriminator.pt'.format(step=self.step))
                    torch.save(self.uncond_discriminator.state_dict(), self.models_loc + 'model_history/{step}/uncond_discriminator.pt'.format(step=self.step))
                    cur_meta_state = {
                        'n_samples':self.n_samples,
                        'steps':self.step,
                        'cond_disc_steps':self.cond_disc_step,
                        'uncond_disc_steps':self.uncond_disc_step,
                        'gen_steps':self.gen_step,
                        'generator_optimizer':self.generator_optimizer.state_dict(),
                        'cond_discriminator_optimizer':self.cond_discriminator_optimizer.state_dict(),
                        'uncond_discriminator_optimizer':self.uncond_discriminator_optimizer.state_dict()
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
            self.cond_discriminator_optimizer.zero_grad()
            self.uncond_discriminator_optimizer.zero_grad()

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

            cond_discrimination = self.cond_discriminator(input_time_states[:, :, 3].view(batch_size, -1, 1), queried)

            uncond_discrimination = self.uncond_discriminator(queried)

            cond_discriminator_loss = self.loss(cond_discrimination, torch.cat([torch.zeros(n_new_gen, 1), torch.ones(n_actual, 1), torch.zeros(n_replay_gen, 1)], 0))
            uncond_discriminator_loss = self.loss(uncond_discrimination, torch.cat([torch.zeros(n_new_gen, 1), torch.ones(n_actual, 1), torch.zeros(n_replay_gen, 1)], 0))

            try:
                assert not torch.isnan(cond_discriminator_loss)
                assert not torch.isnan(uncond_discriminator_loss)
            except AssertionError:
                print(torch.isnan(input_time_states).sum())
                print(torch.isnan(generation).sum())
                print(torch.isnan(cond_discrimination).sum())
                print(torch.isnan(uncond_discrimination).sum())
                for name, param in self.generator.named_parameters():
                    print(name, torch.isnan(param).sum())
                for name, param in self.cond_discriminator.named_parameters():
                    print(name, torch.isnan(param).sum())
                for name, param in self.uncond_discriminator.named_parameters():
                    print(name, torch.isnan(param).sum())
                assert False

            dis_real_mean = cond_discrimination[n_new_gen:n_new_gen + n_actual].mean()
            dis_gen_mean = cond_discrimination[:n_new_gen].mean()
            cond_disc_diff = dis_real_mean - dis_gen_mean

            dis_real_mean = uncond_discrimination[n_new_gen:n_new_gen + n_actual].mean()
            dis_gen_mean = uncond_discrimination[:n_new_gen].mean()
            uncond_disc_diff = dis_real_mean - dis_gen_mean

            self.cond_disc_diff_ema = self.disc_diff_tau * cond_disc_diff.item() + (1 - self.disc_diff_tau) * self.cond_disc_diff_ema
            self.uncond_disc_diff_ema = self.disc_diff_tau * uncond_disc_diff.item() + (1 - self.disc_diff_tau) * self.uncond_disc_diff_ema

            # print("cond diff ema", self.cond_disc_diff_ema)
            # print("uncond diff ema", self.uncond_disc_diff_ema)

            cond_disc_step = self.cond_disc_diff_ema <= 0.02
            uncond_disc_step = self.uncond_disc_diff_ema <= 0.02
            gen_step = not cond_disc_step or not uncond_disc_step

            self.set_learning_rate(cond_disc_step, uncond_disc_step, gen_step)

            if cond_disc_step:
                # print("cond disc step")
                self.cond_disc_step += 1
            cond_discriminator_loss.backward(retain_graph=True)
            self.cond_discriminator_optimizer.step()

            if uncond_disc_step:
                # print("uncond disc step")
                self.uncond_disc_step += 1
            uncond_discriminator_loss.backward(retain_graph=True)
            self.uncond_discriminator_optimizer.step()

            self.generator_optimizer.zero_grad()
            generator_loss = self.loss(cond_discrimination[:n_new_gen], torch.ones(n_new_gen, 1)) + self.loss(uncond_discrimination[:n_new_gen], torch.ones(n_new_gen, 1))
            try:
                assert not torch.isnan(generator_loss)
            except AssertionError:
                print(torch.isnan(input_time_states).sum())
                print(torch.isnan(generation).sum())
                print(torch.isnan(cond_discrimination).sum())
                print(torch.isnan(uncond_discrimination).sum())
                for name, param in self.generator.named_parameters():
                    print(name, torch.isnan(param).sum())
                for name, param in self.cond_discriminator.named_parameters():
                    print(name, torch.isnan(param).sum())
                for name, param in self.uncond_discriminator.named_parameters():
                    print(name, torch.isnan(param).sum())
                assert False

            if gen_step:
                # print("gen step")
                self.gen_step += 1
            generator_loss.backward(retain_graph=False)
            self.generator_optimizer.step()

            for name, param in self.generator.named_parameters():
                try:
                    assert torch.isnan(param).sum() == 0
                except AssertionError:
                    print(name, torch.isnan(param).sum())
                    assert False
            for name, param in self.cond_discriminator.named_parameters():
                try:
                    assert torch.isnan(param).sum() == 0
                except AssertionError:
                    print(name, torch.isnan(param).sum())
                    assert False
            for name, param in self.uncond_discriminator.named_parameters():
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

            print("step: {step}, cond disc steps: {c_disc_step}, uncond disc steps: {u_disc_step}, gen steps: {gen_step}, n_samples: {n_samples}, time: {time}\n \
                cond_disc_diff_ema: {c_disc_diff_ema}, uncond_disc_diff_ema: {u_disc_diff_ema}\n \
                c_dis_target: {c_dis_target}, c_dis_new_gen: {c_dis_new_gen}, c_dis_replay_gen: {c_dis_replay_gen}\n \
                u_dis_target: {u_dis_target}, u_dis_new_gen: {u_dis_new_gen}, u_dis_replay_gen: {u_dis_replay_gen}".format(
                step=self.step,
                c_disc_step=self.cond_disc_step,
                u_disc_step=self.uncond_disc_step,
                gen_step=self.gen_step,
                n_samples=self.n_samples,
                time=round(time.time() - t0, 5),
                c_disc_diff_ema=round(self.cond_disc_diff_ema, 5),
                u_disc_diff_ema=round(self.uncond_disc_diff_ema, 5),
                c_dis_target=round(cond_discrimination[n_new_gen:n_new_gen + n_actual].mean().item(), 5),
                c_dis_new_gen=round(cond_discrimination[:n_new_gen].mean().item(), 5),
                c_dis_replay_gen=round(cond_discrimination[n_new_gen + n_actual:].mean().item(), 5),
                u_dis_target=round(uncond_discrimination[n_new_gen:n_new_gen + n_actual].mean().item(), 5),
                u_dis_new_gen=round(uncond_discrimination[:n_new_gen].mean().item(), 5),
                u_dis_replay_gen=round(uncond_discrimination[n_new_gen + n_actual:].mean().item(), 5),
                ))
            print('-----------------------------------------------------------------------')

            self.step += 1
            self.n_samples += n_experiences
