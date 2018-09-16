import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from collections import namedtuple
from networks import *
from environment import *
from worker import Experience
import redis
import pickle

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


class Optimizer(object):

    def __init__(self, models_loc):
        self.models_loc = models_loc
        # networks
        self.MEN = torch.load(self.models_loc + '/market_encoder.pt').cuda()
        self.AN = torch.load(self.models_loc + '/actor.pt').cuda()
        self.CN = torch.load(self.models_loc + '/critic.pt').cuda()
        self.ON = torch.load(self.models_loc + '/order.pt').cuda()

        # target networks. don't need a target actor network
        self.MEN_ = torch.load(self.models_loc + '/market_encoder.pt').cuda()
        self.CN_ = torch.load(self.models_loc + '/critic.pt').cuda()
        self.ON_ = torch.load(self.models_loc + '/order.pt').cuda()

        self.server = redis.Redis("localhost")
        self.gamma = float(self.server.get("optimizer_gamma").decode("utf-8"))
        self.tau = float(self.server.get("optimizer_tau").decode("utf-8"))
        self.alpha = float(self.server.get("optimizer_alpha").decode("utf-8"))
        self.betas = float(self.server.get("optimizer_betas").decode("utf-8"))
        self.epsilon = float(self.server.get("optimizer_epsilon").decode("utf-8"))
        self.weight_penalty = float(self.server.get("optimizer_weight_penalty").decode("utf-8"))

        self.experience = []
        self.optimizer = optim.Adam(self.MEN.parameters() +
                                    self.AN.parameters() +
                                    self.CN.parameters() +
                                    self.ON.parameters(),
                                    lr=self.alpha,
                                    betas=self.betas,
                                    eps=self.epsilon,
                                    weight_decay=self.weight_penalty)


    def run(self):

        while len(self.experience) != 0:
            # start grads anew
            self.optimizer.zero_grad()

            # get the inputs to the networks in the right form
            batch = Experience(*zip(*self.experience))  # maybe restructure this so that we aren't using the entire experience each batch
            initial_time_states = torch.cat(batch.time_states[:-1], 1).cuda()
            final_time_states = torch.cat(batch.time_states[1:], 1).cuda()
            orders = [orders[i] for orders in batch.orders for i in range(len(orders))]
            queried_amount = torch.cat(batch.queried_amount).cuda()
            orders_actions = torch.cat(batch.orders_actions).cuda()
            place_action = torch.cat(batch.place_action).cuda()
            reward = torch.cat(batch.reward).cuda()
            orders_rewards = torch.cat(batch.orders_rewards).cuda()

            # calculate the market_encoding
            initial_market_encoding = self.MEN.forward(initial_time_states)
            final_market_encoding = self.MEN_.forward(final_time_states)

            # output of actor and critic
            proposed_actions = self.AN.forward(initial_market_encoding)
            expected_value = self.CN_.forward(initial_market_encoding, proposed_actions)[1]

            # backpropagate the gradient for AN
            expected_value.backward()

            # get expected and actual critic values
            expected_advantage, expected_value = self.CN.forward(initial_market_encoding, queried_amount)
            final_advantage, final_value = self.CN_.forward(final_market_encoding, queried_amount)
            actual_Q = reward + ((final_value + final_advantage).max(1)[0].detach() * self.gamma)

            # calculate and backpropagate the critic's loss
            expected_Q = expected_value + expected_advantage.gather(1, torch.Tensor(place_action).long().view(-1, 1))
            critic_loss = F.smooth_l1_loss(expected_Q, actual_Q)
            critic_loss.backward()

            # initial output of orders network
            order_market_encodings = []
            for i, market_encoding in enumerate(initial_market_encoding):
                order_market_encodings.append((market_encoding, len(batch.orders[i])))
            expected_orders_advantage, expected_orders_value = self.ON(order_market_encodings, orders)

            # final output of orders (target) network
            order_market_encodings_ = []
            for i, market_encoding in enumerate(final_market_encoding):
                order_market_encodings_.append((market_encoding), len(batch.orders[i]))
            final_orders_advantage, final_orders_value = self.ON_(order_market_encodings_, orders)

            # set the order targets using the bellman equation
            orders_targets = torch.zeros(len(orders))
            orders_targets = orders_rewards + ((final_orders_value + final_orders_advantage).max(1)[0].detach() * self.gamma)

            # if the orders close, then this is equivalent to the terminal state for the order
            orders_closed_mask = torch.cat([a == 1 for a in orders_actions])
            orders_targets[orders_closed_mask] = orders_rewards[orders_closed_mask]

            # calculate and backpropogate the order's loss
            orders_loss = F.smooth_l1_loss(expected_orders_value + expected_orders_advantage, orders_targets)
            orders_loss.backward()

            # take a step
            self.optimizer.step()

            # update the target networks using a exponential moving average
            for i, param in enumerate(self.MEN_.parameters()):
                param = (self.tau * self.MEN.parameters()[i]) + ((1 - self.tau) * param)

            for i, param in enumerate(self.CN_.parameters()):
                param = (self.tau * self.CN.parameters()[i]) + ((1 - self.tau) * param)

            for i, param in enumerate(self.ON_.parameters()):
                param = (self.tau * self.ON.parameters()[i]) + ((1 - self.tau) * param)

            if int(self.server.get("optimizer_update").decode("utf-8")):
                self.experience = pickle.loads(self.server.get("experience"))
                self.MEN.save(self.models_loc + "/market_encoder.pt")
                self.AN.save(self.models_loc + "/actor.pt")
                self.CN.save(self.models_loc + "/critic.pt")
                self.ON.save(self.models_loc + "/order.pt")
