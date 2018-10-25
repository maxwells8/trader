import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

torch.manual_seed(0)
D_BAR = 5
D_MODEL = 128
N_LSTM_LAYERS = 1
WINDOW = 120
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

    def forward(self, input_market_values, percent_in, spread, device, reset_lstm=True):
        x = None
        if reset_lstm:
            self.hidden = self.init_hidden(input_market_values.size()[1], device)

        x = F.leaky_relu(self.fc1(input_market_values))
        x, self.hidden = self.lstm(x, self.hidden)
        # x = F.leaky_relu(x)
        x = self.hidden
        x = torch.cat([x[-1].view(-1, D_MODEL), percent_in.view(-1, 1), spread.view(-1, 1)], 1)
        return x


class AttentionMarketEncoder(nn.Module):
    """
    TODO:
        - add an output mlp instead of max pooling. this will force a fixed number
        of entities, but should add to its ability to learn.
    """

    def __init__(self):
        super(AttentionMarketEncoder, self).__init__()

        self.fc_bar = nn.Linear(D_BAR, D_MODEL)
        self.fc_in = nn.Linear(1, D_MODEL)
        self.fc_spread = nn.Linear(1, D_MODEL)

        self.N = 2

        self.h = 8
        self.d_k = int(D_MODEL / self.h)
        self.d_v = int(D_MODEL / self.h)

        self.n_entities = WINDOW + 2

        self.WQs = [nn.Linear(D_MODEL, self.d_k, bias=False) for _ in range(self.h)]
        for i, WQ in enumerate(self.WQs):
            self.add_module("WQ" + str(i), WQ)
        self.WKs = [nn.Linear(D_MODEL, self.d_k, bias=False) for _ in range(self.h)]
        for i, WK in enumerate(self.WKs):
            self.add_module("WK" + str(i), WK)
        self.WVs = [nn.Linear(D_MODEL, self.d_v, bias=False) for _ in range(self.h)]
        for i, WV in enumerate(self.WVs):
            self.add_module("WV" + str(i), WV)
        self.WO = nn.Linear(self.h * self.d_v, D_MODEL, bias=False)

        self.fc_out = nn.Linear(D_MODEL, D_MODEL)

        self.fc_final = nn.Linear(WINDOW + 2, 1)

    def forward(self, market_values, percent_in, spread):

        inputs = [F.leaky_relu(self.fc_bar(market_values.view(WINDOW, -1, D_BAR)))]
        inputs += [F.leaky_relu(self.fc_in(percent_in.view(1, -1, 1)))]
        inputs += [F.leaky_relu(self.fc_spread(spread.view(1, -1, 1)))]
        inputs = torch.cat(inputs).transpose(0, 1)

        for _ in range(self.N):
            heads = []
            for i in range(self.h):
                Q = self.WQs[i](inputs)
                Q_mean = Q.mean(dim=2).view(-1, self.n_entities, 1)
                Q_std = Q.std(dim=2).view(-1, self.n_entities, 1)
                Q = (Q - Q_mean) / Q_std
                K = self.WKs[i](inputs)
                K_mean = K.mean(dim=2).view(-1, self.n_entities, 1)
                K_std = K.std(dim=2).view(-1, self.n_entities, 1)
                K = (K - K_mean) / K_std
                V = self.WVs[i](inputs)
                V_mean = V.mean(dim=2).view(-1, self.n_entities, 1)
                V_std = V.std(dim=2).view(-1, self.n_entities, 1)
                V = (V - V_mean) / V_std

                saliencies = torch.bmm(Q, K.transpose(1, 2))
                weights = F.softmax(saliencies / np.sqrt(self.d_k), dim=2)
                head = torch.bmm(weights, V)
                heads.append(head)

            heads = torch.cat(heads, dim=2)
            outputs = F.leaky_relu(self.fc_out(self.WO(heads))) + inputs
            inputs = outputs

        outputs = F.leaky_relu(self.fc_final(outputs.transpose(1, 2))).squeeze()

        return outputs


class Proposer(nn.Module):
    """
    takes a market encoding (which also includes the percentage of balance
    already in a trade) and outputs a value for percentage of the available
    balance to buy and sell
    """

    def __init__(self):
        super(Proposer, self).__init__()
        # this is the lstm's version
        # self.fc1 = nn.Linear(D_MODEL + 2, 2)
        # this is the attention version
        self.fc1 = nn.Linear(D_MODEL, 2)

    def forward(self, market_encoding, exploration_parameter=0):
        # this is the lstm's version
        # x = torch.sigmoid(self.fc1(market_encoding.view(-1, D_MODEL + 2)) + exploration_parameter)
        # this is the attention version
        x = torch.sigmoid(self.fc1(market_encoding.view(-1, D_MODEL)) + exploration_parameter)
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

        # this is the lstm's version
        # self.fc1 = nn.Linear(D_MODEL + self.d_action + 2, D_MODEL)
        # this is the attention version
        self.fc1 = nn.Linear(D_MODEL + self.d_action, D_MODEL)

        # self.actor1 = nn.Linear(D_MODEL, D_MODEL)

        # changing the structure of this so that it won't immediately learn to
        # just not trade
        # self.actor2 = nn.Linear(D_MODEL, 4)
        self.actor2 = nn.Linear(D_MODEL, 2)

        # self.critic1 = nn.Linear(D_MODEL, D_MODEL)
        self.critic2 = nn.Linear(D_MODEL, 1)

    def forward(self, market_encoding, proposed_actions, sigma=1):

        # this is the lstm's version
        # x = F.leaky_relu(self.fc1(torch.cat([market_encoding.view(-1, D_MODEL + 2),
        #                                      proposed_actions.view(-1, self.d_action)], 1)))
        # this is the attention version
        x = F.leaky_relu(self.fc1(torch.cat([market_encoding.view(-1, D_MODEL),
                                             proposed_actions.view(-1, self.d_action)], 1))) + market_encoding.view(-1, D_MODEL)

        # policy = F.leaky_relu(self.actor1(x)) + x
        # policy = self.actor2(policy) * sigma
        # policy = F.softmax(policy, dim=1)
        #
        # critic = F.leaky_relu(self.critic1(x)) + x
        # critic = self.critic2(critic)

        policy = self.actor2(x) * sigma
        policy = F.softmax(policy, dim=1)

        critic = self.critic2(x)

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
AME = AttentionMarketEncoder().cuda()

inputs = torch.cat([torch.randn([1, 2, D_BAR]) for _ in range(3)]).cuda()
percent_in = torch.Tensor([[0.5], [0.75]]).cuda()
spread = torch.Tensor([[1 / 10000], [2 / 10000]]).cuda()

print(AME.forward(inputs, percent_in, spread).size())
"""

"""
ME = MarketEncoder().cuda()
P = Proposer().cuda()
AC = ActorCritic().cuda()

inputs = [torch.randn([1, 1, D_BAR]) for _ in range(512)]

# n = 10
# t0 = time.time()
# for _ in range(n):
market_encoding = ME.forward(torch.cat(inputs).cuda(), torch.Tensor([0.5]).cuda(), 'cuda')
proposed_actions = P.forward(market_encoding)
print(proposed_actions)
proposed_actions = P.forward(market_encoding, 0.1)
print(proposed_actions)
policy, value = AC.forward(market_encoding, proposed_actions)
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
