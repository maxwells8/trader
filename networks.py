import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

torch.manual_seed(0)
D_BAR = 5
D_MODEL = 256
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

    def __init__(self, device='cuda'):
        super(AttentionMarketEncoder, self).__init__()
        self.device = device

        self.fc_bar = nn.Linear(D_BAR, D_MODEL)

        self.N = 6
        self.h = 8
        self.fc_out_middle_size = D_MODEL * 2

        self.d_k = int(D_MODEL / self.h)
        self.d_v = int(D_MODEL / self.h)

        self.n_entities = WINDOW

        self.WQs = nn.ModuleList([nn.Linear(D_MODEL, self.d_k, bias=False) for _ in range(self.h)])
        self.WKs = nn.ModuleList([nn.Linear(D_MODEL, self.d_k, bias=False) for _ in range(self.h)])
        self.WVs = nn.ModuleList([nn.Linear(D_MODEL, self.d_v, bias=False) for _ in range(self.h)])
        self.WO = nn.Linear(self.h * self.d_v, D_MODEL, bias=False)

        self.fc_out1 = nn.ModuleList([nn.Linear(D_MODEL, self.fc_out_middle_size) for _ in range(self.N)])
        self.fc_out2 = nn.ModuleList([nn.Linear(self.fc_out_middle_size, D_MODEL) for _ in range(self.N)])
        self.in_gain = torch.ones(D_MODEL, requires_grad=True)
        self.in_bias = torch.zeros(D_MODEL, requires_grad=True)

        self.fc_final = nn.Linear(WINDOW, 1)

    def forward(self, market_values):
        time_states = []
        for i, time_state in enumerate(market_values):
            if self.device == 'cuda':
                time_states.append(torch.cat([time_state.cuda(), torch.Tensor([(i - WINDOW / 2) / (WINDOW / 2)]).repeat(time_state.size()[0]).view(-1, 1).cuda()], dim=1).view(1, -1, D_BAR))
            else:
                time_states.append(torch.cat([time_state.cpu(), torch.Tensor([(i - WINDOW / 2) / (WINDOW / 2)]).repeat(time_state.size()[0]).view(-1, 1).cpu()], dim=1).view(1, -1, D_BAR))

        time_states = torch.cat(time_states, dim=0)
        inputs_ = [F.leaky_relu(self.fc_bar(time_states.view(WINDOW, -1, D_BAR)))]
        inputs_ = torch.cat(inputs_).transpose(0, 1)
        inputs_mean = inputs_.mean(dim=2).view(-1, self.n_entities, 1)
        inputs_std = inputs_.std(dim=2).view(-1, self.n_entities, 1)
        inputs = (inputs_ - inputs_mean) / (inputs_std + 1e-9)
        inputs = inputs * self.in_gain + self.in_bias

        for j in range(self.N):
            heads = []
            for i in range(self.h):
                Q = self.WQs[i](inputs)
                K = self.WKs[i](inputs)
                V = self.WVs[i](inputs)
                # print(Q, K, V)

                saliencies = torch.bmm(Q, K.transpose(1, 2))
                weights = F.softmax(saliencies / np.sqrt(self.d_k), dim=2)
                # print(j, weights.max(dim=2))
                head = torch.bmm(weights, V)
                heads.append(head)

            heads = torch.cat(heads, dim=2)

            out = F.leaky_relu(self.fc_out1[j](self.WO(heads)))
            out = self.fc_out2[j](out) + inputs

            out_mean = out.mean(dim=2).view(-1, self.n_entities, 1)
            out_std = out.std(dim=2).view(-1, self.n_entities, 1)
            inputs = (out - out_mean) / (out_std + 1e-9)
            inputs = inputs * self.in_gain + self.in_bias

        outputs = F.leaky_relu(self.fc_final(inputs.transpose(1, 2))).view(-1, D_MODEL)
        return outputs


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(D_MODEL + 2, D_MODEL)
        self.gain1 = nn.Parameter(torch.ones(D_MODEL))
        self.bias1 = nn.Parameter(torch.zeros(D_MODEL))

        self.fc2 = nn.Linear(D_MODEL, D_MODEL)
        self.gain2 = nn.Parameter(torch.ones(D_MODEL))
        self.bias2 = nn.Parameter(torch.zeros(D_MODEL))

        self.fc3 = nn.Linear(D_MODEL, 3)

    def forward(self, encoding, spread, log_steps):
        x = torch.cat([encoding.view(-1, D_MODEL), spread.view(-1, 1), log_steps.view(-1, 1)], 1)

        x = F.leaky_relu(self.fc1(x)) + encoding
        mean1 = x.mean(dim=1).view(-1, 1)
        std1 = x.std(dim=1).view(-1, 1)
        x = (x - mean1) / std1
        x = x * self.gain1 + self.bias1

        x = F.leaky_relu(self.fc2(x)) + x
        mean2 = x.mean(dim=1).view(-1, 1)
        std2 = x.std(dim=1).view(-1, 1)
        x = (x - mean2) / std2
        x = x * self.gain2 + self.bias2

        x = self.fc3(x)
        advantage = (x - torch.mean(x, 1).view(-1, 1)) / (torch.sum(x.abs(), 1).view(-1, 1) + 1e-6)

        return advantage


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
        self.actor2 = nn.Linear(D_MODEL, 4)
        # self.actor2 = nn.Linear(D_MODEL, 2)

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


class EncoderToOthers(nn.Module):

    def __init__(self):
        super(EncoderToOthers, self).__init__()
        self.fc1 = nn.Linear(D_MODEL + 2, D_MODEL)

    def forward(self, encoding, spread, percent_in):
        x = torch.cat([encoding.view(-1, D_MODEL), spread.view(-1, 1), percent_in.view(-1, 1)], 1)
        x = F.leaky_relu(self.fc1(x)) + encoding.view(-1, D_MODEL)
        return x


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
