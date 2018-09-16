import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

torch.manual_seed(0)
# torch.cuda.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

"""
Think FIFO
"""
class MarketEncoder(nn.Module):
    """
    goes through each market time step with an lstm, and outputs a 'market encoding'
    """

    def __init__(self, input_dim, d_model, lstm_layers, device):
        super(MarketEncoder, self).__init__()

        self.d_model = d_model
        self.input_dim = input_dim
        self.lstm_layers = lstm_layers
        self.device = device

        self.fc1 = nn.Linear(self.input_dim, self.d_model, device=self.device)
        # self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=lstm_layers, device=self.device)
        self.hidden = self.init_hidden(1)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm_layers, batch_size, self.d_model, device=self.device),
                torch.zeros(self.lstm_layers, batch_size, self.d_model, device=self.device))

    def forward(self, input_market_values, reset_lstm=True):
        x = None
        if reset_lstm:
            self.hidden = self.init_hidden(input_market_values.size()[1])

        x = F.leaky_relu(self.fc1(input_market_values))
        # x = F.leaky_relu(self.fc2(x)) + x
        x, self.hidden = self.lstm(x, self.hidden)
        x = F.leaky_relu(x[-1])
        return x


class Actor(nn.Module):
    """
    takes a market encoding and outputs a value for percentage of the available balance to buy and sell
    """

    def __init__(self, d_model):
        super(Actor, self).__init__()

        self.d_model = d_model

        self.fc1 = nn.Linear(self.d_model, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, 2)

    def forward(self, market_encoding):
        x = F.leaky_relu(self.fc1(market_encoding.view(-1, self.d_model)))
        x = F.leaky_relu(self.fc2(x)) + x
        x = F.sigmoid(self.fc3(x))

        return x


class Critic(nn.Module):
    """
    takes a market encoding and a proposed action, and outputs the advantage and value of buying, selling, and neither
    """

    def __init__(self, d_model, d_action):
        super(Critic, self).__init__()

        self.d_model = d_model
        self.d_action = d_action

        self.fc1 = nn.Linear(self.d_model + 2, self.d_model)

        self.advantage1 = nn.Linear(self.d_model, self.d_model)
        self.advantage2 = nn.Linear(self.d_model, 3)

        self.value1 = nn.Linear(self.d_model, self.d_model)
        self.value2 = nn.Linear(self.d_model, 1)

    def forward(self, market_encoding, action):

        x = F.leaky_relu(self.fc1(torch.cat([market_encoding.view(-1, self.d_model),
                                             action.view(-1, self.d_action)], 1)))

        advantage = F.leaky_relu(self.advantage1(x)) + x
        advantage = self.advantage2(advantage)

        value = F.leaky_relu(self.value1(x)) + x
        value = self.value2(value)

        return advantage, value


class OrderNetwork(nn.Module):
    """
    takes a market encoding and an open order, and outputs the advantage and value of keeping and closing the order

    the order must give the open time information
    """

    def __init__(self, d_model, d_order):
        super(OrderNetwork, self).__init__()

        self.d_model = d_model
        self.d_order = d_order

        self.order_fc1 = nn.Linear(self.d_order, self.d_model)
        self.order_fc2 = nn.Linear(self.d_model, self.d_model)

        self.combine = nn.Linear(2*self.d_model, self.d_model)

        self.advantage1 = nn.Linear(self.d_model, self.d_model)
        self.advantage2 = nn.Linear(self.d_model, 2)

        self.value1 = nn.Linear(self.d_model, self.d_model)
        self.value2 = nn.Linear(self.d_model, 1)

    def forward(self, market_encoding_tuples, orders):
        """
        market_encodings is a list of tuples: [(market_encoding0, n0), ..., (market_encodingk, nk)]
        """
        order_vec = F.leaky_relu(self.order_fc1(orders.view(-1, self.d_order)))
        order_vec = F.leaky_relu(order_vec) + order_vec

        market_encoding = torch.Tensor([], device=str(order_vec.device))
        for METuple in market_encoding_tuples:
            next_ME = METuple[0].view(-1, self.d_model).repeat(METuple[1], 1)
            market_encoding = torch.cat([market_encoding, next_ME], 0)
        combined = F.leaky_relu(self.combine(torch.cat([market_encoding,
                                                       order_vec.view(-1, self.d_model)], 1)))
        # combined = F.leaky_relu(self.combine(torch.cat([market_encoding.repeat(orders.size()[0], 1).view(-1, self.d_model),
        #                                                order_vec.view(-1, self.d_model)], 1)))

        advantage = F.leaky_relu(self.advantage1(combined)) + combined
        advantage = self.advantage2(advantage)

        value = F.leaky_relu(self.value1(combined)) + combined
        value = self.value2(value)

        return advantage, value


"""
d_input = 8
d_model = 256
d_order = 8
ME = MarketEncoder(d_input, d_model, 2)
A = Actor(d_model)
C = Critic(d_model, 2)
O = OrderNetwork(d_model, d_order)

inputs = [torch.randn([1, 1, ME.input_dim]) for _ in range(512)]
orders = torch.randn([64, d_order])

# n = 10
# t0 = time.time()
# for _ in range(n):
market_encoding = ME.forward(torch.cat(inputs))
proposed_actions = A.forward(market_encoding) + (torch.randn(1, 2) * 0.05)
Q_actions = C.forward(market_encoding, proposed_actions)
close = O.forward([(market_encoding, int(len(orders)))], orders)
print(close[0].max(1)[1])
Q_actions[1].backward()
for foo in A.parameters():
    print(foo.grad.size())
torch.save(ME, "models/market_encoder.pt")
torch.save(A, "models/actor.pt")
torch.save(C, "models/critic.pt")
torch.save(O, "models/order.pt")
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
