import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.insert(0, "../")
from collections import namedtuple
import networks
from networks import *
from environment import *
import redis
import msgpack
import math
import gym

# np.random.seed(0)
# torch.manual_seed(0)
torch.set_num_threads(1)

class Worker(object):

    def __init__(self, name, env_name, models_loc, test=False):

        if not test:
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        while True:
            if not test:
                self.state_encoder = FCEncoder().cpu()
                self.actor_critic = ActorCritic().cpu()
                try:
                    self.state_encoder.load_state_dict(torch.load(models_loc + 'state_encoder.pt', map_location='cpu'))
                    self.actor_critic.load_state_dict(torch.load(models_loc + 'actor_critic.pt', map_location='cpu'))
                    break
                except Exception:
                    print("Failed to load models")
                    time.sleep(0.1)
            else:
                self.state_encoder = FCEncoder().cuda()
                self.actor_critic = ActorCritic().cuda()
                try:
                    self.state_encoder.load_state_dict(torch.load(models_loc + 'state_encoder.pt', map_location='cuda'))
                    self.actor_critic.load_state_dict(torch.load(models_loc + 'actor_critic.pt', map_location='cuda'))
                    break
                except Exception:
                    print("Failed to load models")
                    time.sleep(0.1)

        self.state_encoder.eval()
        self.actor_critic.eval()

        self.name = name
        self.models_loc = models_loc

        self.server = redis.Redis("localhost")

        self.states = []
        self.mus = []
        self.dones = []
        self.actions = []
        self.rewards = []

        self.total_actual_reward = 0
        self.reward_tau = float(self.server.get("reward_tau").decode("utf-8"))

        self.trajectory_steps = int(self.server.get("trajectory_steps").decode("utf-8"))

        self.test = test
        self.env_name = env_name
        self.env = gym.make(env_name)

        if self.test:
            self.actor_temp = 1
        else:
            actor_base_temp = float(self.server.get("actor_temp").decode("utf-8"))
            self.actor_temp = np.random.exponential(actor_base_temp / np.log(2))

        self.i_step = 0
        self.steps_since_push = 0
        self.steps_between_experiences = self.trajectory_steps
        self.step_first_push = np.random.uniform(0, self.trajectory_steps)

    def step(self):

        if len(self.dones) > 0 and self.dones[-1] == True:
            policy = torch.ones(1, 2) / 2
            value = torch.zeros(1, 1)
        else:
            encoding = self.state_encoder(torch.Tensor(self.states[-1]))
            policy, value = self.actor_critic(encoding, temp=self.actor_temp)

        if self.test:
            # action = torch.argmax(policy).item()
            action = torch.multinomial(policy, 1).item()
        else:
            action = torch.multinomial(policy, 1).item()

        mu = policy[0, action].item()
        # print("v_t", value)
        # print("mu_t", policy)

        if len(self.dones) == 0 or not self.dones[-1]:
            # print("x_t", np.array(self.states[-1]))
            # print("a_t", action)
            state, reward, done, _ = self.env.step(action)
            # print("r_t", reward)
            # print("x_(t+1)", state)
            # print("done_(t+1)", done)
            # print()
            state = state.tolist() + [np.log(self.i_step + 1e-2)]
        else:
            state = torch.zeros(2).tolist() + [np.log(self.i_step + 1e-2)]
            reward = 0
            done = True

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.mus.append(mu)
        self.dones.append(done)

        self.total_actual_reward += reward

        if self.test:
            self.env.render()
            reward_ema = self.server.get("test_reward_ema")
            reward_emsd = self.server.get("test_reward_emsd")
            if reward_ema != None:
                reward_ema = float(reward_ema.decode("utf-8"))
                reward_emsd = float(reward_emsd.decode("utf-8"))
            else:
                reward_ema = 0
                reward_emsd = 0

            print("step: {s} \
            \naction: {a} \
            \npolicy: {p} \
            \nvalue: {v} \
            \nrewards: {r} \
            \nreward_ema: {ema} \
            \nreward_emsd: {emsd} \
            \nenv: {env} \n".format(s=self.i_step,
                                    a=action,
                                    p=[round(policy_, 5) for policy_ in policy[0].tolist()],
                                    v=round(value.item(), 5),
                                    r=round(self.total_actual_reward, 5),
                                    ema=round(reward_ema, 5),
                                    emsd=round(reward_emsd, 5),
                                    env=self.env_name))

            if done:
                return

        if len(self.states) == self.trajectory_steps + 1:
            del self.states[0]
            del self.rewards[0]
            del self.dones[0]
        if len(self.actions) == self.trajectory_steps + 1:
            del self.actions[0]
            del self.mus[0]

        if (self.i_step > self.step_first_push) and (self.steps_since_push > self.steps_between_experiences) and (not self.test) and len(self.actions) == self.trajectory_steps:

            if self.dones[0] is True:
                return

            experience = Experience(
            states=self.states,
            mus=self.mus,
            done=self.dones,
            place_actions=self.actions,
            rewards=self.rewards
            )

            experience = msgpack.packb(experience, use_bin_type=True)
            self.add_to_replay_buffer(experience)
            n_experiences = self.server.llen("experience")
            if n_experiences > 0:
                try:
                    loc = np.random.randint(0, n_experiences)
                    ref = self.server.lindex("experience", loc)
                    self.server.linsert("experience", "before", ref, experience)
                except redis.exceptions.DataError:
                    self.server.lpush("experience", experience)
            else:
                self.server.lpush("experience", experience)
            self.steps_since_push = 1

        else:
            self.steps_since_push += 1


        self.i_step += 1
        if self.i_step > 200:
            return

        self.step()


    def run(self):
        t0 = time.time()

        self.states.append(torch.Tensor(self.env.reset()).tolist() + [np.log(self.i_step + 1e-2)])
        self.step()

        print(("time: {time}, "
                "rewards: {reward} %, "
                "temp: {actor_temp}").format(
                    time=round(time.time()-t0, 2),
                    reward=self.total_actual_reward,
                    actor_temp=round(self.actor_temp, 3)
                )
        )


    def add_to_replay_buffer(self, compressed_experience):
        # try:
        #     loc = np.random.randint(0, replay_size) if replay_size > 0 else 0
        #     ref = self.server.lindex("replay_buffer", loc)
        #     self.server.linsert("replay_buffer", "before", ref, compressed_experience)
        # except redis.exceptions.DataError:
        self.server.lpush("replay_buffer", compressed_experience)

        max_replay_size = int(self.server.get("replay_buffer_size").decode("utf-8"))
        replay_size = self.server.llen("replay_buffer")
        while replay_size - 1 > max_replay_size:
            replay_size = self.server.llen("replay_buffer")
            try:
                loc = replay_size - 1
                ref = self.server.lindex("replay_buffer", loc)
                self.server.lrem("replay_buffer", -1, ref)
            except Exception as e:
                pass

Experience = namedtuple('Experience', ('states',
                                       'mus',
                                       'done',
                                       'place_actions',
                                       'rewards'))
