import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

torch.manual_seed(0)
D_BAR = 6
D_MODEL = 256
N_LSTM_LAYERS = 2
# torch.cuda.manual_seed(0)
# torch.set_default_tensor_type(torch.cuda.FloatTensor)


class MarketEncoder(nn.Module):
    """
    goes through each market time step with an lstm, and outputs a
    'market encoding' which includes the percentage of balance already in a
    trade
    """

    def __init__(self, device='cuda'):
        super(MarketEncoder, self).__init__()

        self.fc1 = nn.Linear(D_BAR, D_MODEL)
        self.lstm = nn.LSTM(input_size=D_MODEL, hidden_size=D_MODEL, num_layers=N_LSTM_LAYERS)
        self.hidden = self.init_hidden(1, device)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(N_LSTM_LAYERS, batch_size, D_MODEL, device=device),
                torch.zeros(N_LSTM_LAYERS, batch_size, D_MODEL, device=device))

    def forward(self, input_market_values, percent_in, device, reset_lstm=True):
        x = None
        if reset_lstm:
            self.hidden = self.init_hidden(input_market_values.size()[1], device)

        x = F.leaky_relu(self.fc1(input_market_values))
        x, self.hidden = self.lstm(x, self.hidden)
        x = F.leaky_relu(torch.cat([x[-1].view(-1, D_MODEL), percent_in.view(-1, 1)], 1))
        return x


class Proposer(nn.Module):
    """
    takes a market encoding (which also includes the percentage of balance
    already in a trade) and outputs a value for percentage of the available
    balance to buy and sell
    """

    def __init__(self):
        super(Proposer, self).__init__()

        self.fc1 = nn.Linear(D_MODEL + 1, D_MODEL)
        self.fc2 = nn.Linear(D_MODEL, D_MODEL)
        self.fc3 = nn.Linear(D_MODEL, 2)

    def forward(self, market_encoding):
        x = F.leaky_relu(self.fc1(market_encoding.view(-1, D_MODEL + 1)))
        x = F.leaky_relu(self.fc2(x)) + x
        x = F.sigmoid(self.fc3(x))
        return x


class ActorCritic(nn.Module):
    """
    takes a market encoding (which also includes the percentage of balance
    already in a trade) and a proposed action, and outputs the policy (buy,
    sell, keep), and the value of the current state
    """

    def __init__(self):
        super(ActorCritic, self).__init__()

        self.d_action = 2

        self.fc1 = nn.Linear(D_MODEL + self.d_action + 1, D_MODEL)

        self.actor1 = nn.Linear(D_MODEL, D_MODEL)
        self.actor2 = nn.Linear(D_MODEL, 3)

        self.critic1 = nn.Linear(D_MODEL, D_MODEL)
        self.critic2 = nn.Linear(D_MODEL, 1)

    def forward(self, market_encoding, proposed_actions):

        x = F.leaky_relu(self.fc1(torch.cat([market_encoding.view(-1, D_MODEL + 1),
                                             proposed_actions.view(-1, self.d_action)], 1)))

        policy = F.leaky_relu(self.actor1(x)) + x
        policy = F.softmax(self.actor2(policy), dim=1)

        critic = F.leaky_relu(self.critic1(x)) + x
        critic = self.critic2(critic)

        return policy, critic


# class OrderNetwork(nn.Module):
#     """
#     takes a market encoding and an open order, and outputs the advantage and value of keeping and closing the order
#
#     the order must give the open time information
#     """
#
#     def __init__(self, d_model, d_order):
#         super(OrderNetwork, self).__init__()
#
#         self.d_model = d_model
#         self.d_order = d_order
#
#         self.order_fc1 = nn.Linear(self.d_order, self.d_model)
#         self.order_fc2 = nn.Linear(self.d_model, self.d_model)
#
#         self.combine = nn.Linear(2*self.d_model, self.d_model)
#
#         self.advantage1 = nn.Linear(self.d_model, self.d_model)
#         self.advantage2 = nn.Linear(self.d_model, 2)
#
#         self.value1 = nn.Linear(self.d_model, self.d_model)
#         self.value2 = nn.Linear(self.d_model, 1)
#
#     def forward(self, market_encoding_tuples, orders):
#         """
#         market_encodings is a list of tuples: [(market_encoding0, n0), ..., (market_encodingk, nk)]
#         """
#         order_vec = F.leaky_relu(self.order_fc1(orders.view(-1, self.d_order)))
#         order_vec = F.leaky_relu(order_vec) + order_vec
#
#         market_encoding = torch.Tensor([], device=str(order_vec.device))
#         for METuple in market_encoding_tuples:
#             next_ME = METuple[0].view(-1, self.d_model).repeat(METuple[1], 1)
#             market_encoding = torch.cat([market_encoding, next_ME], 0)
#         combined = F.leaky_relu(self.combine(torch.cat([market_encoding,
#                                                        order_vec.view(-1, self.d_model)], 1)))
#         # combined = F.leaky_relu(self.combine(torch.cat([market_encoding.repeat(orders.size()[0], 1).view(-1, self.d_model),
#         #                                                order_vec.view(-1, self.d_model)], 1)))
#
#         advantage = F.leaky_relu(self.advantage1(combined)) + combined
#         advantage = self.advantage2(advantage)
#
#         value = F.leaky_relu(self.value1(combined)) + combined
#         value = self.value2(value)
#
#         return advantage, value


"""
ME = MarketEncoder().cuda()
P = Proposer().cuda()
AC = ActorCritic().cuda()

inputs = [torch.randn([1, 1, D_BAR]) for _ in range(512)]

# n = 10
# t0 = time.time()
# for _ in range(n):
market_encoding = ME.forward(torch.cat(inputs).cuda(), torch.Tensor([0.5]).cuda(), 'cuda')
proposed_actions = P.forward(market_encoding) + (torch.randn(1, 2).cuda() * 0.05)
policy, value = AC.forward(market_encoding, proposed_actions)
print(proposed_actions, policy, value)
print(torch.log(policy)[0, 1]*value)
(torch.log(policy)[0, 1]*value).backward()

torch.save(ME, "models/market_encoder.pt")
torch.save(P, "models/proposer.pt")
torch.save(AC, "models/actor_critic.pt")
# print((time.time() - t0) / n)
"""
"""
ME = MarketEncoder(8, 256, 2)
inputs = [torch.randn([1, 1, ME.input_dim]) for _ in range(400)]
n = 10
t0 = time.time()
for _ in range(n):
    out = ME.forward(inputs)
print((time.time() - t0) / n)
"""

"""
### like 30 minutes of work on a relation reasoning model that i gave up on

class Bot(nn.Module):

    def __init__(self, d_price, d_order, n_heads, d_model, n_blocks):
        super(Bot, self).__init__()

        self.d_price = d_price
        self.d_order = d_order
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.d_k = self.d_model / self.n_heads
        self.d_v = self.d_model / self.n_heads
        self.Q = None
        self.K = None
        self.V = None

        self.linear_price_transformation = nn.Linear(self.d_price, self.d_model)

        self.relational_mlp1 = nn.Linear(self.d_model, self.d_model)
        self.relational_mlp2 = nn.Linear(self.d_model, self.d_model)

        self.shared_mlp1 = nn.Linear(self.d_model, self.d_model)
        self.shared_mlp2 = nn.Linear(self.d_model, self.d_model)
        self.shared_mlp3 = nn.Linear(self.d_model, self.d_model)
        self.shared_mlp4 = nn.Linear(self.d_model, self.d_model)

        # each order's value of keeping / closing
        self.order_mlp1 = nn.Linear(self.d_order + self.d_model, self.d_model)
        self.order_mlp2 = nn.Linear(self.d_model, self.d_model)
        self.order_mlp3 = nn.Linear(self.d_model, self.d_model)
        self.order_mlp4 = nn.Linear(self.d_model, self.d_model)

        self.order_advantage_mlp1 = nn.Linear(self.d_model, self.d_model)
        self.order_advantage_mlp2 = nn.Linear(self.d_model, 2)

        self.order_value_mlp1 = nn.Linear(self.d_model, self.d_model)
        self.order_value_mlp2 = nn.Linear(self.d_model, 1)

        # placing an order actor critic
        self.place_mlp1 = nn.Linear(self.d_model, self.d_model)
        self.place_mlp2 = nn.Linear(self.d_model, self.d_model)
        self.place_mlp3 = nn.Linear(self.d_model, self.d_model)
        self.place_mlp4 = nn.Linear(self.d_model, self.d_model)

        self.place_quantity_mlp1 = nn.Linear(self.d_model, self.d_model)
        self.place_quantity_mlp2 = nn.Linear(self.d_model, self.d_model)
        self.place_quantity_mlp3 = nn.Linear(self.d_model, self.d_model)
        self.place_quantity_mlp4 = nn.Linear(self.d_model, 2)

        self.place_critic_mlp1 = nn.Linear(2 + self.d_model, self.d_model)
        self.place_critic_mlp2 = nn.Linear(self.d_model, self.d_model)
        self.place_critic_mlp3 = nn.Linear(self.d_model, self.d_model)
        self.place_critic_mlp4 = nn.Linear(self.d_model, self.d_model)

        self.place_critic_advantage_mlp1 = nn.Linear(self.d_model, self.d_model)
        self.place_critic_advantage_mlp2 = nn.Linear(self.d_model, 3)

        self.place_critic_value_mlp1 = nn.Linear(self.d_model, self.d_model)
        self.place_critic_value_mlp2 = nn.Linear(self.d_model, 1)

    def forward(self, prices, orders):
        pass
"""
