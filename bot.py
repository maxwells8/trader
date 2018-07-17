import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


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

