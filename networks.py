import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import time

torch.manual_seed(0)
D_BAR = 4
D_MODEL = 512
WINDOW = 180
P_DROPOUT = 0
# torch.cuda.manual_seed(0)
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

class FCEncoder(nn.Module):
    def __init__(self):
        super(FCEncoder, self).__init__()

        self.initial_fc = nn.Linear(3, D_MODEL)
        self.initial_bn = nn.BatchNorm1d(D_MODEL)
        self.n_layers = 2
        self.fcs = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_layers)])

    def forward(self, input):
        x = input.view(-1, 3)
        x = self.initial_fc(x)
        x = self.initial_bn(x)
        x = F.leaky_relu(x)

        for i in range(self.n_layers):
            if i % 2 == 0:
                res = x
                x = self.fcs[i](x)
                x = self.bns[i](x)
                x = F.leaky_relu(x)
            else:
                x = self.fcs[i](x)
                x = self.bns[i](x)
                x = x + res
                x = F.leaky_relu(x)

        return x


class CNNRelationalEncoder(nn.Module):
    """
    not finished
    """
    def __init__(self):
        super(CNNRelationalEncoder, self).__init__()

        self.n_channels = [int(D_MODEL / 4), int(D_MODEL / 2), D_MODEL]
        self.kernel_sizes = [7, 5, 3]
        self.pool_kernel_size = 2
        self.n_tot_entities = 128
        self.d_e = int(D_MODEL / 2)
        self.d_v = int(D_MODEL / 2)
        self.d_u = int(D_MODEL / 2)

        self.n_pos_entities = WINDOW
        for k in self.kernel_sizes:
            self.n_pos_entities = int(math.floor(self.n_pos_entities - k + 1))
            self.n_pos_entities = int(math.floor((self.n_pos_entities - self.pool_kernel_size) / self.pool_kernel_size + 1))
        assert self.n_pos_entities < self.n_tot_entities

        self.conv1 = nn.Conv1d(in_channels=D_BAR,
                                out_channels=self.n_channels[0],
                                kernel_size=self.kernel_sizes[0])
        self.conv2 = nn.Conv1d(in_channels=self.n_channels[0],
                                out_channels=self.n_channels[1],
                                kernel_size=self.kernel_sizes[1])
        self.conv3 = nn.Conv1d(in_channels=self.n_channels[1],
                                out_channels=self.n_channels[2],
                                kernel_size=self.kernel_sizes[2])

        self.max_pool = nn.MaxPool1d(self.pool_kernel_size)

        self.fc_pos_vertices = nn.Linear(self.n_channels[-1] + 1, self.d_v)

        self.fc_e1 = nn.Linear(self.d_e + 2 * self.d_v + self.d_u, 2 * self.d_e)
        self.fc_e2 = nn.Linear(2 * self.d_e, self.d_e)
        self.fc_v1 = nn.Linear(self.d_e + self.d_v + self.d_u, 2 * self.d_v)
        self.fc_v2 = nn.Linear(2 * self.d_v, self.d_v)
        self.fc_u1 = nn.Linear(self.d_e + self.d_v + self.d_u, 2 * self.d_u)
        self.fc_u2 = nn.Linear(2 * self.d_u, self.d_u)

        # vertex dimensions as (N, n_vertices, D_MODEL)
        self.init_non_pos_vertices = torch.rand(self.n_tot_entities - self.n_pos_entities, self.d_v, requires_grad=True)
        self.init_u = torch.rand(self.d_u, requires_grad=True)


class CNNResEncoder(nn.Module):
    """
    not finished
    """
    def __init__(self):
        super(CNNResEncoder, self).__init__()

        self.n_blocks = [1, 2, 2, 2]
        self.n_channels = [int(D_MODEL / 8), int(D_MODEL / 4), int(D_MODEL / 2), D_MODEL]
        self.kernel_sizes = [3, 3, 3, 3]

        self.sequence = nn.Sequential()
        for i in range(len(self.n_blocks)):
            name = 'stage' + str(i)
            n_blocks = self.n_blocks[i]
            in_channels = self.n_channels[i-1] if i > 0 else D_BAR
            out_channels = self.n_channels[i]
            kernel_size = self.kernel_sizes[i]
            self.sequence.add_module(
                name,
                self._make_stage(n_blocks, in_channels, out_channels, kernel_size)
            )

    def _make_stage(self, n_blocks, in_channels, out_channels, kernel_size):
        stage = nn.Sequential()
        for i in range(n_blocks):
            name = 'block' + str(i)
            in_channels_ = in_channels
            out_channels_ = in_channels if i != n_blocks - 1 else out_channels
            stage.add_module(
                name,
                self._make_block(in_channels_, out_channels_, kernel_size)
            )

        return stage

    def _make_block(self, in_channels, out_channels, kernel_size):
        block = nn.Sequential()
        return block


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.n_channels = [int(D_MODEL / 8), int(D_MODEL / 4), int(D_MODEL / 2), D_MODEL]
        self.kernel_sizes = [3, 3, 3, 3]
        self.pool_kernel_size = 2

        self.conv1 = nn.Conv1d(in_channels=D_BAR,
                                out_channels=self.n_channels[0],
                                kernel_size=self.kernel_sizes[0])
        self.conv1_bn = nn.BatchNorm1d(self.n_channels[0])

        self.conv2 = nn.Conv1d(in_channels=self.n_channels[0],
                                out_channels=self.n_channels[1],
                                kernel_size=self.kernel_sizes[1])
        self.conv2_bn = nn.BatchNorm1d(self.n_channels[1])

        self.conv3 = nn.Conv1d(in_channels=self.n_channels[1],
                                out_channels=self.n_channels[2],
                                kernel_size=self.kernel_sizes[2])
        self.conv3_bn = nn.BatchNorm1d(self.n_channels[2])

        self.conv4 = nn.Conv1d(in_channels=self.n_channels[2],
                                out_channels=self.n_channels[3],
                                kernel_size=self.kernel_sizes[3])
        self.conv4_bn = nn.BatchNorm1d(self.n_channels[3])

        self.pool = nn.AvgPool1d(self.pool_kernel_size)

        self.L_in = WINDOW
        for k in self.kernel_sizes:
            self.L_in = math.floor(self.L_in - k + 1)
            self.L_in = math.floor((self.L_in - self.pool_kernel_size) / self.pool_kernel_size + 1)
        self.L_in = int(self.L_in)

        self.n_fc_layers = 1
        begin_dim = self.L_in * self.n_channels[-1]
        end_dim = D_MODEL
        # scale down the layer size linearly from begin_dim to end_dim
        self.fc_layers = nn.ModuleList([nn.Linear(int((end_dim - begin_dim) * n / self.n_fc_layers + begin_dim), int((end_dim - begin_dim) * (n + 1) / self.n_fc_layers + begin_dim)) for n in range(self.n_fc_layers)])
        self.fc_bn = nn.ModuleList([nn.BatchNorm1d(int((end_dim - begin_dim) * (n + 1) / self.n_fc_layers + begin_dim)) for n in range(self.n_fc_layers)])

    def forward(self, market_values):
        time_states = market_values.transpose(0, 1).transpose(1, 2)
        batch_size = time_states.size()[0]

        x = self.conv1(time_states)
        x = self.conv1_bn(x)
        x = self.pool(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.pool(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.pool(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.pool(x)
        x = F.leaky_relu(x)

        x = x.view(batch_size, -1)

        for i in range(self.n_fc_layers):
            x = self.fc_layers[i](x)
            x = self.fc_bn[i](x)
            x = F.leaky_relu(x)

        return x


class LSTMCNNEncoder(nn.Module):
    def __init__(self):
        super(LSTMCNNEncoder, self).__init__()

        self.n_lstm_layers = 1
        self.lstm_in_dim = int(D_MODEL / 8)
        self.lstm_hidden_dim = int(D_MODEL / 4)
        self.n_init_channels = int(D_MODEL / 4)
        self.n_channels = [int(D_MODEL / 2), D_MODEL]
        self.kernel_sizes = [3, 3]
        self.pool_kernel_size = 2


        self.bar_to_lstm = nn.Sequential(
                                nn.Linear(D_BAR, self.lstm_in_dim),
                                nn.LeakyReLU()
                            )

        self.lstm = nn.LSTM(input_size=self.lstm_in_dim,
                            hidden_size=self.lstm_hidden_dim,
                            num_layers=self.n_lstm_layers,
                            batch_first=True)

        self.lstm_to_conv = nn.Sequential(
                                nn.Linear(self.lstm_hidden_dim, self.n_init_channels - D_BAR),
                                nn.LeakyReLU()
                            )


        self.conv1 = nn.Conv1d(in_channels=self.n_init_channels,
                                out_channels=self.n_channels[0],
                                kernel_size=self.kernel_sizes[0])
        self.conv1_bn = nn.BatchNorm1d(self.n_channels[0])

        self.conv2 = nn.Conv1d(in_channels=self.n_channels[0],
                                out_channels=self.n_channels[1],
                                kernel_size=self.kernel_sizes[1])
        self.conv2_bn = nn.BatchNorm1d(self.n_channels[1])

        self.pool = nn.AvgPool1d(self.pool_kernel_size)


        self.L_in = WINDOW
        for k in self.kernel_sizes:
            self.L_in = math.floor(self.L_in - k + 1)
            self.L_in = math.floor((self.L_in - self.pool_kernel_size) / self.pool_kernel_size + 1)
        self.L_in = int(self.L_in)

        self.conv_out_dim = self.L_in * self.n_channels[-1]
        begin_dim = self.conv_out_dim
        end_dim = D_MODEL
        self.n_fc_layers = 1
        # scale down the layer size linearly from begin_dim to end_dim
        self.fc_layers = nn.ModuleList([nn.Linear(int((end_dim - begin_dim) * n / self.n_fc_layers + begin_dim), int((end_dim - begin_dim) * (n + 1) / self.n_fc_layers + begin_dim)) for n in range(self.n_fc_layers)])
        self.fc_bn = nn.ModuleList([nn.BatchNorm1d(int((end_dim - begin_dim) * (n + 1) / self.n_fc_layers + begin_dim)) for n in range(self.n_fc_layers)])

    def forward(self, market_values):
        time_states = []
        for time_state in market_values:
            time_states.append(time_state.view(-1, 1, D_BAR))

        # concatting such that the dim is (batch_size, WINDOW, D_BAR)
        time_states = torch.cat(time_states, dim=1)

        x = self.bar_to_lstm(time_states)
        x, _ = self.lstm(x)
        x = self.lstm_to_conv(x)
        x = torch.cat([x, time_states], dim=2)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.pool(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.pool(x)
        x = F.leaky_relu(x)

        x = x.squeeze().view(-1, self.conv_out_dim)

        for i in range(self.n_fc_layers):
            x = self.fc_layers[i](x)
            x = self.fc_bn[i](x)
            x = F.leaky_relu(x)

        return x


class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()

        self.n_bar_layers = 2
        self.n_lstm_layers = 1


        self.initial_layer = nn.Linear(D_BAR, D_MODEL)
        # self.initial_bn = nn.BatchNorm1d(D_MODEL)
        self.bar_layers_1 = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_bar_layers)])
        # self.bar_bns_1 = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_bar_layers)])
        self.dropout = nn.Dropout(p=P_DROPOUT)

        self.lstm_1 = nn.LSTM(input_size=D_MODEL,
                            hidden_size=D_MODEL,
                            num_layers=self.n_lstm_layers)


        self.bar_layers_2 = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_bar_layers)])
        # self.bar_bns_2 = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_bar_layers)])

        self.lstm_2 = nn.LSTM(input_size=D_MODEL,
                            hidden_size=D_MODEL,
                            num_layers=self.n_lstm_layers)

    def forward(self, market_values):
        """
        market_values of size (WINDOW, batch_size, D_BAR)
        """
        x = market_values.transpose(0, 1)
        x = self.initial_layer(x)
        x = x.transpose(1, 2)
        # x = self.initial_bn(x)
        x = F.leaky_relu(x)
        for i in range(self.n_bar_layers):
            if i % 2 == 0:
                res = x
            x = self.dropout(x)
            x = x.transpose(1, 2)
            x = self.bar_layers_1[i](x)
            x = x.transpose(1, 2)
            if i % 2 == 1:
                x = x + res
            x = F.leaky_relu(x)
            # x = self.bar_bns_1[i](x)

        # x = x.transpose(0, 1)
        # x = x.transpose(0, 2)
        x = x.permute(2, 0, 1)
        x = self.dropout(x)
        x, _ = self.lstm_1(x)

        # x = x.transpose(0, 1)
        # x = x.transpose(1, 2)
        x = x.permute(1, 2, 0)

        x = F.leaky_relu(x)
        for i in range(self.n_bar_layers):
            if i % 2 == 0:
                res = x
            x = self.dropout(x)
            x = x.transpose(1, 2)
            x = self.bar_layers_2[i](x)
            x = x.transpose(1, 2)
            if i % 2 == 1:
                x = x + res
            x = F.leaky_relu(x)
            # x = self.bar_bns_2[i](x)

        # x = x.transpose(0, 1)
        # x = x.transpose(0, 2)
        x = x.permute(2, 0, 1)
        x = self.dropout(x)
        x, _ = self.lstm_2(x)

        return x[-1]

class LSTMVariationalEncoder(nn.Module):
    def __init__(self):
        super(LSTMVariationalEncoder, self).__init__()

        self.n_bar_layers = 0
        self.n_lstm_layers = 1
        self.d_encoding = 8


        self.initial_layer = nn.Linear(D_BAR, D_MODEL)
        # self.initial_bn = nn.BatchNorm1d(D_MODEL)
        self.bar_layers_1 = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_bar_layers)])
        # self.bar_bns_1 = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_bar_layers)])
        self.dropout = nn.Dropout(p=P_DROPOUT)

        self.lstm_1 = nn.LSTM(input_size=D_MODEL,
                            hidden_size=D_MODEL,
                            num_layers=self.n_lstm_layers)


        self.bar_layers_2 = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_bar_layers)])
        # self.bar_bns_2 = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_bar_layers)])

        self.lstm_2 = nn.LSTM(input_size=D_MODEL,
                            hidden_size=D_MODEL,
                            num_layers=self.n_lstm_layers)

        self.fc_mean = nn.Linear(D_MODEL, self.d_encoding)
        self.fc_std = nn.Linear(D_MODEL, self.d_encoding)


    def forward(self, market_values):
        """
        market_values of size (WINDOW, batch_size, D_BAR)
        """
        x = market_values.transpose(0, 1)
        x = self.initial_layer(x)
        x = x.transpose(1, 2)
        # x = self.initial_bn(x)
        x = F.leaky_relu(x)
        for i in range(self.n_bar_layers):
            if i % 2 == 0:
                res = x
            x = self.dropout(x)
            x = x.transpose(1, 2)
            x = self.bar_layers_1[i](x)
            x = x.transpose(1, 2)
            if i % 2 == 1:
                x = x + res
            # x = self.bar_bns_1[i](x)
            x = F.leaky_relu(x)

        # x = x.transpose(0, 1)
        # x = x.transpose(0, 2)
        x = x.permute(2, 0, 1)
        x = self.dropout(x)
        x, _ = self.lstm_1(x)

        # x = x.transpose(0, 1)
        # x = x.transpose(1, 2)
        x = x.permute(1, 2, 0)
        x = F.leaky_relu(x)
        for i in range(self.n_bar_layers):
            if i % 2 == 0:
                res = x
            x = self.dropout(x)
            x = x.transpose(1, 2)
            x = self.bar_layers_2[i](x)
            x = x.transpose(1, 2)
            if i % 2 == 1:
                x = x + res
            # x = self.bar_bns_2[i](x)
            x = F.leaky_relu(x)

        # x = x.transpose(0, 1)
        # x = x.transpose(0, 2)
        x = x.permute(2, 0, 1)
        x = self.dropout(x)
        x, _ = self.lstm_2(x)

        x = x[-1]
        means = self.fc_mean(x)
        stds = torch.abs(self.fc_std(x)) + 1e-5
        standard_normal_sample = torch.normal(torch.zeros_like(means), torch.ones_like(stds))

        return standard_normal_sample, means, stds


class LSTMDiscriminator(nn.Module):
    def __init__(self):
        super(LSTMDiscriminator, self).__init__()

        self.n_bar_layers = 0
        self.n_lstm_layers = 1
        self.n_combined_layers = 2


        self.initial_layer = nn.Linear(D_BAR, D_MODEL)
        # self.initial_bn = nn.BatchNorm1d(D_MODEL)
        self.bar_layers_1 = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_bar_layers)])
        # self.bar_bns_1 = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_bar_layers)])
        self.dropout = nn.Dropout(p=P_DROPOUT)

        self.lstm_1 = nn.LSTM(input_size=D_MODEL,
                            hidden_size=D_MODEL,
                            num_layers=self.n_lstm_layers)


        self.bar_layers_2 = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_bar_layers)])
        # self.bar_bns_2 = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_bar_layers)])

        self.lstm_2 = nn.LSTM(input_size=D_MODEL,
                            hidden_size=D_MODEL,
                            num_layers=self.n_lstm_layers)

        self.combined_initial_fc = nn.Linear(D_MODEL + D_BAR, D_MODEL)
        # self.combined_initial_bn = nn.BatchNorm1d(D_MODEL)
        self.combined_layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_combined_layers)])
        # self.combined_bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_combined_layers)])

        self.fc_out = nn.Linear(D_MODEL, 1)


    def forward(self, market_values, query):
        """
        market_values of size (WINDOW, batch_size, D_BAR)
        """
        x = market_values.transpose(0, 1)
        x = self.initial_layer(x)
        x = x.transpose(1, 2)
        # x = self.initial_bn(x)
        x = F.leaky_relu(x)
        for i in range(self.n_bar_layers):
            if i % 2 == 0:
                res = x
            x = self.dropout(x)
            x = x.transpose(1, 2)
            x = self.bar_layers_1[i](x)
            x = x.transpose(1, 2)
            if i % 2 == 1:
                x = x + res
            # x = self.bar_bns_1[i](x)
            x = F.leaky_relu(x)

        x = x.permute(2, 0, 1)
        x = self.dropout(x)
        x, _ = self.lstm_1(x)

        x = x.permute(1, 2, 0)
        x = F.leaky_relu(x)
        for i in range(self.n_bar_layers):
            if i % 2 == 0:
                res = x
            x = self.dropout(x)
            x = x.transpose(1, 2)
            x = self.bar_layers_2[i](x)
            x = x.transpose(1, 2)
            if i % 2 == 1:
                x = x + res
            # x = self.bar_bns_2[i](x)
            x = F.leaky_relu(x)

        x = x.permute(2, 0, 1)
        x = self.dropout(x)
        x, _ = self.lstm_2(x)

        x = x[-1]

        x = torch.cat([x, query.view(-1, D_BAR)], dim=1)
        x = self.combined_initial_fc(x)
        # x = self.combined_initial_bn(x)
        x = F.leaky_relu(x)

        for i in range(self.n_combined_layers):
            if i % 2 == 0:
                res = x
            x = self.dropout(x)
            x = self.combined_layers[i](x)
            if i % 2 == 1:
                x = x + res
            # x = self.combined_bns[i](x)
            x = F.leaky_relu(x)

        p = torch.sigmoid(self.fc_out(x))

        return p


# class Generator(nn.Module):
#
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         self.d_encoding = 8
#         self.fc_init = nn.Linear(self.d_encoding, D_MODEL)
#
#         self.n_res_layers = 2
#         self.res_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_res_layers)])
#
#         self.final_layer = nn.Linear(D_MODEL, D_BAR)
#
#     def forward(self, encoding):
#         x = encoding.view(-1, self.d_encoding)
#         x = self.fc_init(x)
#
#         for res_layer in self.res_layers:
#             x = res_layer(x)
#
#         x = self.final_layer(x)
#         return x


class MarketEncoder(nn.Module):

    def __init__(self, device='cuda'):
        super(MarketEncoder, self).__init__()

        self.fc_bar = nn.Linear(D_BAR, D_MODEL)
        self.lstm = nn.LSTM(input_size=D_MODEL, hidden_size=D_MODEL, num_layers=2)
        self.fc_out = nn.Linear(D_MODEL, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, market_values):
        """
        market_values of size (WINDOW, batch_size, D_BAR)
        """
        x = market_values.transpose(0, 1)
        x = F.leaky_relu(self.fc_bar(x))
        x = x.transpose(0, 1)
        x, _ = self.lstm(x)
        x = x[-1]
        x = self.softmax(self.fc_out(x))
        return x


class AttentionMarketEncoder(nn.Module):

    def __init__(self):
        super(AttentionMarketEncoder, self).__init__()

        # self.N = D_MODEL // 128
        # self.h = D_MODEL // 128
        self.N = 4
        self.h = 8
        self.fc_out_middle_size = 4 * D_MODEL

        self.d_k = D_MODEL // self.h
        self.d_v = D_MODEL // self.h
        self.n_latent_entities = 1

        self.n_entities = WINDOW + self.n_latent_entities

        self.pos_enc = nn.Parameter(torch.normal(torch.zeros(WINDOW, D_MODEL), torch.ones(WINDOW, D_MODEL) / math.sqrt(2)))
        self.fc_bar = nn.Linear(D_BAR, D_MODEL)
        self.output_parameter = nn.Parameter(torch.normal(torch.zeros(self.n_latent_entities, D_MODEL),
                                                        torch.ones(self.n_latent_entities, D_MODEL) / math.sqrt(2)))
        self.layer_norm_init = nn.LayerNorm(D_MODEL)

        self.WQs = nn.ModuleList([nn.Linear(D_MODEL, self.h * self.d_k) for _ in range(self.N)])
        self.WKs = nn.ModuleList([nn.Linear(D_MODEL, self.h * self.d_k) for _ in range(self.N)])
        self.WVs = nn.ModuleList([nn.Linear(D_MODEL, self.h * self.d_v) for _ in range(self.N)])
        self.WOs = nn.ModuleList(nn.Linear(self.h * self.d_v, D_MODEL) for _ in range(self.N))

        self.fc_out1 = nn.ModuleList([nn.Linear(D_MODEL, self.fc_out_middle_size) for _ in range(self.N)])
        self.fc_out2 = nn.ModuleList([nn.Linear(self.fc_out_middle_size, D_MODEL) for _ in range(self.N)])

        for i in range(self.N):
            nn.init.normal_(self.WQs[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WQs[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WKs[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WKs[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WVs[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WVs[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WOs[i].weight, mean=0, std=math.sqrt(2 / (self.N * self.h * self.d_v)))
            nn.init.normal_(self.WOs[i].bias, mean=0, std=math.sqrt(2 / (self.N * self.h * self.d_v)))
            nn.init.normal_(self.fc_out1[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.fc_out1[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.fc_out2[i].weight, mean=0, std=math.sqrt(2 / (self.N * self.fc_out_middle_size)))
            nn.init.normal_(self.fc_out2[i].bias, mean=0, std=math.sqrt(2 / (self.N * self.fc_out_middle_size)))

        self.attention_layer_norms = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(self.N)])
        self.fc_layer_norms = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(self.N)])

        self.dropout = nn.Dropout(P_DROPOUT)

        self.fc_final = nn.Linear(D_MODEL, 3)
        self.softmax = nn.Softmax(1)

    def forward(self, market_values):
        batch_size = market_values.size()[1]
        device = market_values.device

        inputs = market_values.transpose(0, 1)
        inputs = self.fc_bar(inputs)
        inputs = inputs + self.pos_enc.repeat(batch_size, 1, 1)
        inputs = torch.cat([inputs, self.output_parameter.repeat(batch_size, 1, 1)], dim=1)
        inputs = self.layer_norm_init(inputs)

        for i_N in range(self.N):
            # inputs of size (batch_size, self.n_entities, D_MODEL)
            residual = inputs

            Q = self.WQs[i_N](inputs).view(batch_size, self.n_entities, self.h, self.d_k)
            K = self.WKs[i_N](inputs).view(batch_size, self.n_entities, self.h, self.d_k)
            V = self.WVs[i_N](inputs).view(batch_size, self.n_entities, self.h, self.d_v)
            Q = Q.permute(0, 2, 1, 3).contiguous().view(-1, self.n_entities, self.d_k)
            K = K.permute(0, 2, 1, 3).contiguous().view(-1, self.n_entities, self.d_k)
            V = V.permute(0, 2, 1, 3).contiguous().view(-1, self.n_entities, self.d_v)

            attention = torch.bmm(Q, K.transpose(1, 2))
            attention = F.softmax(attention / math.sqrt(self.d_k), dim=2)
            # print(attention)
            attention = self.dropout(attention)
            out = torch.bmm(attention, V)

            out = out.view(batch_size, self.h, self.n_entities, self.d_v)
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, self.n_entities, self.h * self.d_v)
            out = self.WOs[i_N](out)
            out = self.dropout(out)
            out = self.attention_layer_norms[i_N](out + residual)

            residual = out
            out = F.leaky_relu(self.fc_out1[i_N](out))
            out = self.fc_out2[i_N](out)
            out = self.dropout(out)
            out = self.fc_layer_norms[i_N](out + residual)

            inputs = out

        final = self.softmax(self.fc_final(out[:, -1].view(batch_size, D_MODEL)))

        return final


class AttentionVariationalEncoder(nn.Module):

    def __init__(self):
        super(AttentionVariationalEncoder, self).__init__()

        self.N = 8
        self.h = 8
        self.fc_out_middle_size = 4 * D_MODEL
        self.n_latent_entities = 2
        self.d_encoding = 8

        self.d_k = D_MODEL // self.h
        self.d_v = D_MODEL // self.h

        self.n_entities = WINDOW + self.n_latent_entities

        self.pos_enc = nn.Parameter(torch.normal(torch.zeros(WINDOW, D_MODEL), torch.ones(WINDOW, D_MODEL) / math.sqrt(2)))
        self.fc_bar = nn.Sequential(nn.Linear(D_BAR, D_MODEL),
                                    nn.LeakyReLU(),
                                    nn.Linear(D_MODEL, D_MODEL),
                                    nn.Dropout(P_DROPOUT),
                                    nn.LayerNorm(D_MODEL))
        self.output_parameter = nn.Parameter(torch.normal(torch.zeros(self.n_latent_entities, D_MODEL),
                                                        torch.ones(self.n_latent_entities, D_MODEL) / math.sqrt(2)))
        self.layer_norm_init = nn.LayerNorm(D_MODEL)

        self.WQs = nn.ModuleList([nn.Linear(D_MODEL, self.h * self.d_k) for _ in range(self.N)])
        self.WKs = nn.ModuleList([nn.Linear(D_MODEL, self.h * self.d_k) for _ in range(self.N)])
        self.WVs = nn.ModuleList([nn.Linear(D_MODEL, self.h * self.d_v) for _ in range(self.N)])
        self.WOs = nn.ModuleList(nn.Linear(self.h * self.d_v, D_MODEL) for _ in range(self.N))

        self.fc_out1 = nn.ModuleList([nn.Linear(D_MODEL, self.fc_out_middle_size) for _ in range(self.N)])
        self.fc_out2 = nn.ModuleList([nn.Linear(self.fc_out_middle_size, D_MODEL) for _ in range(self.N)])

        for i in range(self.N):
            nn.init.normal_(self.WQs[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WQs[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WKs[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WKs[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WVs[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WVs[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WOs[i].weight, mean=0, std=math.sqrt(2 / (self.N * self.h * self.d_v)))
            nn.init.normal_(self.WOs[i].bias, mean=0, std=math.sqrt(2 / (self.N * self.h * self.d_v)))
            nn.init.normal_(self.fc_out1[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.fc_out1[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.fc_out2[i].weight, mean=0, std=math.sqrt(2 / (self.N * self.fc_out_middle_size)))
            nn.init.normal_(self.fc_out2[i].bias, mean=0, std=math.sqrt(2 / (self.N * self.fc_out_middle_size)))

        self.attention_layer_norms = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(self.N)])
        self.fc_layer_norms = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(self.N)])

        self.dropout = nn.Dropout(P_DROPOUT)

        self.fc_mean = nn.Linear(D_MODEL, self.d_encoding)
        self.fc_std = nn.Linear(D_MODEL, self.d_encoding)

    def forward(self, market_values):
        batch_size = market_values.size()[1]
        device = market_values.device

        inputs = market_values.transpose(0, 1)
        inputs = self.fc_bar(inputs)
        inputs = inputs + self.pos_enc.repeat(batch_size, 1, 1)
        inputs = torch.cat([inputs, self.output_parameter.repeat(batch_size, 1, 1)], dim=1)
        inputs = self.layer_norm_init(inputs)

        for i_N in range(self.N):
            # inputs of size (batch_size, self.n_entities, D_MODEL)
            residual = inputs

            Q = self.WQs[i_N](inputs).view(batch_size, self.n_entities, self.h, self.d_k)
            K = self.WKs[i_N](inputs).view(batch_size, self.n_entities, self.h, self.d_k)
            V = self.WVs[i_N](inputs).view(batch_size, self.n_entities, self.h, self.d_v)
            Q = Q.permute(0, 2, 1, 3).contiguous().view(-1, self.n_entities, self.d_k)
            K = K.permute(0, 2, 1, 3).contiguous().view(-1, self.n_entities, self.d_k)
            V = V.permute(0, 2, 1, 3).contiguous().view(-1, self.n_entities, self.d_v)

            attention = torch.bmm(Q, K.transpose(1, 2))
            attention = F.softmax(attention / math.sqrt(self.d_k), dim=2)
            attention = self.dropout(attention)
            # print(i_N, attention)
            out = torch.bmm(attention, V)

            out = out.view(batch_size, self.h, self.n_entities, self.d_v)
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, self.n_entities, self.h * self.d_v)
            out = self.WOs[i_N](out)
            out = self.dropout(out)
            out = self.attention_layer_norms[i_N](out + residual)

            residual = out
            out = F.leaky_relu(self.fc_out1[i_N](out))
            out = self.fc_out2[i_N](out)
            out = self.dropout(out)
            out = self.fc_layer_norms[i_N](out + residual)

            inputs = out

        means = self.fc_mean(out[:, -1])
        stds = torch.abs(self.fc_std(out[:, -2])) + 1e-5
        standard_normal_sample = torch.normal(torch.zeros(batch_size, self.d_encoding), torch.ones(batch_size, self.d_encoding))

        return standard_normal_sample, means, stds


class AttentionDiscriminator(nn.Module):

    def __init__(self):
        super(AttentionDiscriminator, self).__init__()

        self.N = 8
        self.h = 8
        self.fc_out_middle_size = 4 * D_MODEL

        self.d_k = D_MODEL // self.h
        self.d_v = D_MODEL // self.h

        self.n_entities = WINDOW + 1

        self.pos_enc = nn.Parameter(torch.normal(torch.zeros(WINDOW, D_MODEL), torch.ones(WINDOW, D_MODEL) / math.sqrt(2)))
        self.fc_bar = nn.Sequential(nn.Linear(D_BAR, D_MODEL),
                                    nn.LeakyReLU(),
                                    nn.Linear(D_MODEL, D_MODEL),
                                    nn.Dropout(P_DROPOUT),
                                    nn.LayerNorm(D_MODEL))
        self.fc_disc = nn.Sequential(nn.Linear(D_BAR, D_MODEL),
                                    nn.LeakyReLU(),
                                    nn.Linear(D_MODEL, D_MODEL),
                                    nn.Dropout(P_DROPOUT),
                                    nn.LayerNorm(D_MODEL))
        self.layer_norm_init = nn.LayerNorm(D_MODEL)

        self.WQs = nn.ModuleList([nn.Linear(D_MODEL, self.h * self.d_k) for _ in range(self.N)])
        self.WKs = nn.ModuleList([nn.Linear(D_MODEL, self.h * self.d_k) for _ in range(self.N)])
        self.WVs = nn.ModuleList([nn.Linear(D_MODEL, self.h * self.d_v) for _ in range(self.N)])
        self.WOs = nn.ModuleList(nn.Linear(self.h * self.d_v, D_MODEL) for _ in range(self.N))

        self.fc_out1 = nn.ModuleList([nn.Linear(D_MODEL, self.fc_out_middle_size) for _ in range(self.N)])
        self.fc_out2 = nn.ModuleList([nn.Linear(self.fc_out_middle_size, D_MODEL) for _ in range(self.N)])

        for i in range(self.N):
            nn.init.normal_(self.WQs[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WQs[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WKs[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WKs[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WVs[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WVs[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.WOs[i].weight, mean=0, std=math.sqrt(2 / (self.N * self.h * self.d_v)))
            nn.init.normal_(self.WOs[i].bias, mean=0, std=math.sqrt(2 / (self.N * self.h * self.d_v)))
            nn.init.normal_(self.fc_out1[i].weight, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.fc_out1[i].bias, mean=0, std=math.sqrt(2 / (self.N * D_MODEL)))
            nn.init.normal_(self.fc_out2[i].weight, mean=0, std=math.sqrt(2 / (self.N * self.fc_out_middle_size)))
            nn.init.normal_(self.fc_out2[i].bias, mean=0, std=math.sqrt(2 / (self.N * self.fc_out_middle_size)))

        self.attention_layer_norms = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(self.N)])
        self.fc_layer_norms = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(self.N)])

        self.dropout = nn.Dropout(P_DROPOUT)

        self.fc_final = nn.Linear(D_MODEL, 1)

    def forward(self, market_values, queried):
        batch_size = market_values.size()[1]
        device = market_values.device

        inputs = market_values.transpose(0, 1)
        inputs = self.fc_bar(inputs)
        inputs = inputs + self.pos_enc.repeat(batch_size, 1, 1)
        discriminatee = self.fc_disc(queried).view(batch_size, 1, D_MODEL)
        inputs = torch.cat([inputs, discriminatee], dim=1)
        inputs = self.layer_norm_init(inputs)

        for i_N in range(self.N):
            # inputs of size (batch_size, self.n_entities, D_MODEL)
            residual = inputs

            Q = self.WQs[i_N](inputs).view(batch_size, self.n_entities, self.h, self.d_k)
            K = self.WKs[i_N](inputs).view(batch_size, self.n_entities, self.h, self.d_k)
            V = self.WVs[i_N](inputs).view(batch_size, self.n_entities, self.h, self.d_v)
            Q = Q.permute(0, 2, 1, 3).contiguous().view(-1, self.n_entities, self.d_k)
            K = K.permute(0, 2, 1, 3).contiguous().view(-1, self.n_entities, self.d_k)
            V = V.permute(0, 2, 1, 3).contiguous().view(-1, self.n_entities, self.d_v)

            attention = torch.bmm(Q, K.transpose(1, 2))
            attention = F.softmax(attention / math.sqrt(self.d_k), dim=2)
            attention = self.dropout(attention)
            out = torch.bmm(attention, V)

            out = out.view(batch_size, self.h, self.n_entities, self.d_v)
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, self.n_entities, self.h * self.d_v)
            out = self.WOs[i_N](out)
            out = self.dropout(out)
            out = self.attention_layer_norms[i_N](out + residual)

            residual = out
            out = F.leaky_relu(self.fc_out1[i_N](out))
            out = self.fc_out2[i_N](out)
            out = self.dropout(out)
            out = self.fc_layer_norms[i_N](out + residual)

            inputs = out

        final = torch.sigmoid(self.fc_final(out[:, -1]))

        return final


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.d_encoding = 16
        self.fc_init = nn.Linear(self.d_encoding, D_MODEL)
        self.bn_init = nn.BatchNorm1d(D_MODEL)

        # self.n_layers = D_MODEL // 128
        self.n_layers = 4
        self.layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_layers)])

        for i in range(self.n_layers):
            nn.init.normal_(self.layers[i].weight, mean=0, std=math.sqrt(2 / (self.n_layers * D_MODEL)))
            nn.init.normal_(self.layers[i].bias, mean=0, std=math.sqrt(2 / (self.n_layers * D_MODEL)))

        self.final_layer = nn.Linear(D_MODEL, D_BAR)

        self.dropout = nn.Dropout(P_DROPOUT)

    def forward(self, encoding):
        x = encoding.view(-1, self.d_encoding)
        x = self.fc_init(x)
        x = self.bn_init(x)
        x = F.leaky_relu(x)

        for i in range(self.n_layers):
            if i % 2 == 0:
                res = x
                x = self.layers[i](x)
                x = self.dropout(x)
                x = self.bns[i](x)
                x = F.leaky_relu(x)
            else:
                x = self.layers[i](x)
                x = self.dropout(x)
                x = x + res
                x = self.bns[i](x)
                x = F.leaky_relu(x)

        x = self.final_layer(x)
        return x


class ClassifierDecoder(nn.Module):

    def __init__(self):
        super(ClassifierDecoder, self).__init__()
        self.fc1 = nn.Linear(D_MODEL + 3, D_MODEL)
        self.fc2 = nn.Linear(D_MODEL, D_MODEL)
        self.fc3 = nn.Linear(D_MODEL, D_MODEL)
        self.fc_final = nn.Linear(D_MODEL, 3)

    def forward(self, encoding, spread, std, confidence_interval):
        x = torch.cat([
            encoding.view(-1, D_MODEL),
            (spread + 1e-6).log().view(-1, 1),
            (std + 1e-6).log().view(-1, 1),
            (confidence_interval + 1e-6).log().view(-1, 1)
        ], 1)

        x = F.leaky_relu(self.fc1(x) + encoding.view(-1, D_MODEL))
        x = F.leaky_relu(self.fc2(x) + x)
        x = F.leaky_relu(self.fc3(x) + x)
        x = self.fc_final(x)
        x = F.softmax(x, dim=1)

        return x


class Proposer(nn.Module):
    """
    takes a market encoding (which also includes the percentage of balance
    already in a trade) and outputs a value for percentage of the available
    balance to buy and sell
    """

    def __init__(self):
        super(Proposer, self).__init__()
        d_out = 2
        self.fc1 = nn.Linear(D_MODEL, D_MODEL)
        self.fc1_gain = nn.Parameter(torch.zeros(D_MODEL))
        self.fc1_bias = nn.Parameter(torch.zeros(D_MODEL))

        self.fc2 = nn.Linear(D_MODEL, d_out)

    def forward(self, market_encoding, exploration_parameter=0):
        x = self.fc1(market_encoding.view(-1, D_MODEL)) + market_encoding.view(-1, D_MODEL)
        mean = x.mean(dim=1).view(-1, 1)
        std = x.std(dim=1).view(-1, 1)
        x = (x - mean) / std
        x = (1 + self.fc1_gain) * x + self.fc1_bias
        x = F.leaky_relu(x)

        x = self.fc2(x) + exploration_parameter
        x = torch.sigmoid(x)
        return x


class ProbabilisticProposer(nn.Module):

    def __init__(self):
        super(ProbabilisticProposer, self).__init__()
        self.d_out = 2
        self.d_mixture = 4

        self.dist_n_layers = 2
        self.dist_layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.dist_n_layers)])
        self.dist_bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.dist_n_layers)])
        self.dropout = nn.Dropout(P_DROPOUT)
        self.layer_mu = nn.Linear(D_MODEL, self.d_out * self.d_mixture)
        self.layer_sigma = nn.Linear(D_MODEL, self.d_out * self.d_mixture)
        self.layer_w = nn.Linear(D_MODEL, self.d_out * self.d_mixture)

    def forward(self, market_encoding, return_params=False, temp=None):
        if temp == None:
            temp = {"w":1, "mu":1, "sigma":1}

        pdf = market_encoding.view(-1, D_MODEL)
        for i in range(self.dist_n_layers):
            if i % 2 == 0:
                res = pdf
                pdf = self.dropout(pdf)
                pdf = self.dist_layers[i](pdf)
                pdf = self.dist_bns[i](pdf)
                pdf = self.dropout(pdf)
                pdf = F.leaky_relu(pdf)
            else:
                pdf = self.dist_layers[i](pdf)
                pdf = self.dist_bns[i](pdf)
                pdf = pdf + res
                pdf = F.leaky_relu(pdf)

        w = self.layer_w(pdf).view(-1, self.d_out, self.d_mixture)
        w = w / temp["w"]
        w = torch.softmax(w, dim=2)

        mu = self.layer_mu(pdf).view(-1, self.d_out, self.d_mixture)
        mu = mu / temp["mu"]

        sigma = self.layer_sigma(pdf).view(-1, self.d_out, self.d_mixture)
        sigma = sigma / temp["sigma"]
        sigma = torch.abs(sigma) + 1e-5

        x = self.sample(w, mu, sigma)
        p_x = self.p(x.detach(), w, mu, sigma)

        if return_params:
            return x, p_x, w, mu, sigma
        else:
            return x, p_x

    def p(self, x, w, mu, sigma):
        x = x.view(-1, self.d_out, 1)
        x = torch.min(torch.max(x,
                                torch.zeros(1, device=x.device) + 1e-6),
                      torch.ones(1, device=x.device) - 1e-6)
        # probability density function for logit normal
        p_x = (1 / (torch.sqrt(2 * math.pi * sigma ** 2))) * \
                (1 / (x * (1 - x))) * \
                torch.exp(-(torch.log(x / (1 - x)) - mu) ** 2 / (2 * sigma ** 2))
        # getthing the combined probability
        p_x = (w * p_x).sum(2)
        return p_x.view(-1, self.d_out)

    def inverse_cdf(self, p, mu, sigma):
        # sampling from a logit normal distribution
        x = torch.sigmoid(torch.erfinv(p * 2 - 1) * sigma * math.sqrt(2) + mu)
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        return x

    def sample(self, w, mu, sigma):
        i = torch.multinomial(w.view(-1, self.d_mixture), 1).view(w.size()[0], self.d_out, 1)
        p = torch.rand(w.size()[0], self.d_out, 1, device=w.device)
        samples = self.inverse_cdf(p, mu.gather(2, i), sigma.gather(2, i))
        sample = (w * samples).sum(2).view(-1, self.d_out)
        return sample


class ProposerGate(nn.Module):

    def __init__(self):
        super(ProposerGate, self).__init__()
        self.d_dist = 2

        self.dropout = nn.Dropout(P_DROPOUT)
        self.layer_init = nn.Linear(D_MODEL + 2 * self.d_dist, D_MODEL)
        self.bn_init = nn.BatchNorm1d(D_MODEL)
        self.layer_out = nn.Linear(D_MODEL, 2)

    def forward(self, market_encoding, cur_amounts, proposed_amounts, temp=1):
        p = torch.cat([market_encoding.view(-1, D_MODEL), cur_amounts.view(-1, self.d_dist), proposed_amounts.view(-1, self.d_dist)], dim=1)
        p = self.layer_init(p)
        p = self.dropout(p)
        p = F.leaky_relu(p)
        p = self.layer_out(p)
        p = p / temp
        p = F.softmax(p, dim=1)

        return p


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc_init = nn.Linear(WINDOW * D_BAR + 2, D_MODEL)
        # self.ln_init = nn.LayerNorm(D_MODEL)

        self.n_res_layers = 4
        self.res_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_res_layers)])

    def forward(self, market_values, spread, percent_in, trade_open):
        """
        market_values of size (batch_size, WINDOW, D_BAR)
        """
        batch_size = market_values.size()[0]

        means = market_values[:, :, 3].contiguous().view(batch_size, -1).mean(1)
        stds = market_values[:, :, 3].contiguous().view(batch_size, -1).std(1)
        market_values_ = (market_values - means.view(batch_size, 1, 1)) / (stds.view(batch_size, 1, 1) + 1e-9)
        spread_ = spread / (stds + 1e-9)
        trade_open_ = (trade_open - means) / (stds + 1e-9)

        # x = torch.cat([market_values_.view(batch_size, -1), spread_.view(batch_size, 1), percent_in.view(batch_size, 1), trade_open_.view(batch_size, 1)], dim=1)
        x = torch.cat([market_values_.view(batch_size, -1), spread_.view(batch_size, 1), percent_in.view(batch_size, 1)], dim=1)

        x = self.fc_init(x)
        # x = self.ln_init(x)
        x = F.leaky_relu(x)

        for layer in self.res_layers:
            x = layer(x)

        return x


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()

        self.d_action = 4

        self.n_actor_layers = 1
        self.actor_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_actor_layers)])
        self.final_actor = nn.Linear(D_MODEL, self.d_action)
        self.actor_softmax = nn.Softmax(dim=1)

        self.n_critic_layers = 1
        self.critic_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_critic_layers)])
        self.final_critic = nn.Linear(D_MODEL, 1)

    def forward(self, market_encoding, temp=1):

        x = market_encoding

        policy = x
        for layer in self.actor_layers:
            policy = layer(policy)
        policy = self.final_actor(policy) / temp
        policy = self.actor_softmax(policy)

        critic = x
        for layer in self.critic_layers:
            critic = layer(critic)
        critic = self.final_critic(critic)

        return policy, critic


class EncoderToOthers(nn.Module):

    def __init__(self):
        super(EncoderToOthers, self).__init__()
        self.initial_layer = nn.Linear(D_MODEL + 2, D_MODEL)
        # self.initial_bn = nn.BatchNorm1d(D_MODEL)

        self.n_layers = 0
        self.layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_layers)])
        # self.bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL) for _ in range(self.n_layers)])
        for i in range(self.n_layers):
            nn.init.normal_(self.layers[i].weight, mean=0, std=math.sqrt(1 / (self.n_layers * D_MODEL)))
            nn.init.normal_(self.layers[i].bias, mean=0, std=math.sqrt(2 / (self.n_layers * D_MODEL)))

        self.dropout = nn.Dropout(P_DROPOUT)

    def forward(self, encoding, spread, percent_in):
        x = torch.cat([
                    encoding.view(-1, D_MODEL),
                    spread.view(-1, 1),
                    percent_in.view(-1, 1)
                    ], 1)

        x = self.initial_layer(x)
        # x = self.initial_bn(x)
        x = F.leaky_relu(x)
        for i in range(self.n_layers):
            if i % 2 == 0:
                res = x
            x = self.dropout(x)
            x = self.layers[i](x)
            # x = self.bns[i](x)
            if i % 2 == 1:
                x = x + res
            x = F.leaky_relu(x)

        return x


class FCResLayer(nn.Module):
    def __init__(self):
        super(FCResLayer, self).__init__()

        self.intermediate_size = D_MODEL * 2
        self.fc_init = nn.Linear(D_MODEL, self.intermediate_size)
        self.fc_final = nn.Linear(self.intermediate_size, D_MODEL)

        # self.ln = nn.LayerNorm(D_MODEL)

        self.dropout = nn.Dropout(P_DROPOUT)

        for param in [self.fc_init, self.fc_final]:
            nn.init.normal_(param.weight, mean=0, std=0.02)
            nn.init.normal_(param.bias, mean=0, std=0.02)

    def forward(self, input):

        x = self.dropout(input)
        x = self.fc_init(x)
        x = F.leaky_relu(x)

        x = self.dropout(x)
        x = self.fc_final(x)

        x = x + input
        # x = self.ln(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.h = 8
        self.d_k = D_MODEL // self.h
        self.d_v = D_MODEL // self.h

        self.WQ = nn.Linear(D_MODEL, self.h * self.d_k)
        self.WK = nn.Linear(D_MODEL, self.h * self.d_k)
        self.WV = nn.Linear(D_MODEL, self.h * self.d_v)
        self.WO = nn.Linear(self.h * self.d_v, D_MODEL)

        self.attention_layer_norm = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(P_DROPOUT)

        self.fc_res_layer = FCResLayer()

        for param in [self.WQ, self.WK, self.WV, self.WO]:
            nn.init.normal_(param.weight, mean=0, std=0.02)
            nn.init.normal_(param.bias, mean=0, std=0.02)

    def forward(self, input):
        """
        input of size (batch_size, sequence_length, D_MODEL)
        """
        batch_size, sequence_length = input.size()[:2]

        Q = self.WQ(input).view(batch_size, sequence_length, self.h, self.d_k)
        K = self.WK(input).view(batch_size, sequence_length, self.h, self.d_k)
        V = self.WV(input).view(batch_size, sequence_length, self.h, self.d_v)
        Q = Q.permute(0, 2, 1, 3).contiguous().view(-1, sequence_length, self.d_k)
        K = K.permute(0, 2, 1, 3).contiguous().view(-1, sequence_length, self.d_k)
        V = V.permute(0, 2, 1, 3).contiguous().view(-1, sequence_length, self.d_v)

        attention = torch.bmm(Q, K.transpose(1, 2))
        attention = F.softmax(attention / math.sqrt(self.d_k), dim=2)
        attention = self.dropout(attention)
        out = torch.bmm(attention, V)

        out = out.view(batch_size, self.h, sequence_length, self.d_v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, self.h * self.d_v)
        out = self.WO(out)
        out = self.dropout(out)
        out = out + input
        out = self.attention_layer_norm(out)

        out = self.fc_res_layer(out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.h = 8
        self.d_k = D_MODEL // self.h
        self.d_v = D_MODEL // self.h

        self.WQ_output = nn.Linear(D_MODEL, self.h * self.d_k)
        self.WK_output = nn.Linear(D_MODEL, self.h * self.d_k)
        self.WV_output = nn.Linear(D_MODEL, self.h * self.d_v)
        self.WO_output = nn.Linear(self.h * self.d_v, D_MODEL)

        self.WQ_input = nn.Linear(D_MODEL, self.h * self.d_k)
        self.WK_input = nn.Linear(D_MODEL, self.h * self.d_k)
        self.WV_input = nn.Linear(D_MODEL, self.h * self.d_v)
        self.WO_input = nn.Linear(self.h * self.d_v, D_MODEL)

        self.output_attention_layer_norm = nn.LayerNorm(D_MODEL)
        self.input_attention_layer_norm = nn.LayerNorm(D_MODEL)

        self.dropout = nn.Dropout(P_DROPOUT)

        self.fc_res_layer = FCResLayer()

        for param in [self.WQ_output, self.WK_output, self.WV_output, self.WO_output,
                    self.WQ_input, self.WK_input, self.WV_input, self.WO_input]:
            nn.init.normal_(param.weight, mean=0, std=0.02)
            nn.init.normal_(param.bias, mean=0, std=0.02)

    def forward(self, input, output):
        """
        input of size (batch_size, sequence_length, D_MODEL)
        """
        device = input.device
        batch_size = input.size()[0]
        input_sequence_length = input.size()[1]
        output_sequence_length = output.size()[1]

        residual = output
        Q = self.WQ_output(output).view(batch_size, output_sequence_length, self.h, self.d_k)
        K = self.WK_output(output).view(batch_size, output_sequence_length, self.h, self.d_k)
        V = self.WV_output(output).view(batch_size, output_sequence_length, self.h, self.d_v)
        Q = Q.permute(0, 2, 1, 3).contiguous().view(-1, output_sequence_length, self.d_k)
        K = K.permute(0, 2, 1, 3).contiguous().view(-1, output_sequence_length, self.d_k)
        V = V.permute(0, 2, 1, 3).contiguous().view(-1, output_sequence_length, self.d_v)

        attention_out = torch.bmm(Q, K.transpose(1, 2))
        attention_out = F.softmax(attention_out / math.sqrt(self.d_k), dim=2)
        mask = torch.triu(torch.ones((output_sequence_length, output_sequence_length), device=device, dtype=torch.uint8), diagonal=1)
        attention_out.masked_fill(mask.repeat(batch_size * self.h, 1, 1), -np.inf)
        attention_out = self.dropout(attention_out)
        attention_out = torch.bmm(attention_out, V)

        attention_out = attention_out.view(batch_size, self.h, output_sequence_length, self.d_v)
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(batch_size, output_sequence_length, self.h * self.d_v)
        attention_out = self.WO_output(attention_out)
        attention_out = self.dropout(attention_out)
        attention_out = attention_out + residual
        attention_out = self.output_attention_layer_norm(attention_out)


        residual = attention_out
        Q = self.WQ_input(attention_out).view(batch_size, output_sequence_length, self.h, self.d_k)
        K = self.WK_input(input).view(batch_size, input_sequence_length, self.h, self.d_k)
        V = self.WV_input(input).view(batch_size, input_sequence_length, self.h, self.d_v)
        Q = Q.permute(0, 2, 1, 3).contiguous().view(-1, output_sequence_length, self.d_k)
        K = K.permute(0, 2, 1, 3).contiguous().view(-1, input_sequence_length, self.d_k)
        V = V.permute(0, 2, 1, 3).contiguous().view(-1, input_sequence_length, self.d_v)

        attention_out = torch.bmm(Q, K.transpose(1, 2))
        attention_out = F.softmax(attention_out / math.sqrt(self.d_k), dim=2)
        attention_out = self.dropout(attention_out)
        attention_out = torch.bmm(attention_out, V)

        attention_out = attention_out.view(batch_size, self.h, output_sequence_length, self.d_v)
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(batch_size, output_sequence_length, self.h * self.d_v)
        attention_out = self.WO_input(attention_out)
        attention_out = self.dropout(attention_out)
        attention_out = attention_out + residual
        attention_out = self.input_attention_layer_norm(attention_out)


        out = self.fc_res_layer(attention_out)

        return out


class BarEmbedder(nn.Module):
    def __init__(self):
        super(BarEmbedder, self).__init__()

        self.init_layer = nn.Linear(1, D_MODEL)

        self.n_res_layers = 0
        self.res_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_res_layers)])

        for param in [self.init_layer]:
            nn.init.normal_(param.weight, mean=0, std=0.02)
            nn.init.normal_(param.bias, mean=0, std=0.02)

    def forward(self, input):

        x = self.init_layer(input)

        for layer in self.res_layers:
            x = layer(x)

        return x


class GenEncoder(nn.Module):
    def __init__(self):
        super(GenEncoder, self).__init__()

        self.n_attn_layers = 4
        self.attn_layers = nn.ModuleList([EncoderLayer() for _ in range(self.n_attn_layers)])

    def forward(self, input):

        for layer in self.attn_layers:
            input = layer(input)

        return input


class GenDecoder(nn.Module):
    def __init__(self):
        super(GenDecoder, self).__init__()

        self.n_attn_layers = 4
        self.attn_layers = nn.ModuleList([DecoderLayer() for _ in range(self.n_attn_layers)])

        self.n_fc_res_enc_layers = 1
        self.fc_res_enc_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_fc_res_enc_layers)])

        self.d_encoding = 32
        self.fc_enc = nn.Linear(D_MODEL, self.d_encoding)
        # self.enc_ln = nn.LayerNorm(self.d_encoding)
        # self.enc_noise_ln = nn.LayerNorm(self.d_encoding)

        self.fc_out_init = nn.Linear(self.d_encoding, D_MODEL)
        self.n_fc_res_out_layers = 1
        self.fc_res_out_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_fc_res_out_layers)])
        self.fc_out_final = nn.Linear(D_MODEL, 1)

        for param in [self.fc_enc, self.fc_out_init, self.fc_out_final]:
            nn.init.normal_(param.weight, mean=0, std=0.02)
            nn.init.normal_(param.bias, mean=0, std=0.02)

    def forward(self, input, output):

        for layer in self.attn_layers:
            output = layer(input, output)

        for layer in self.fc_res_enc_layers:
            output = layer(output)

        enc = self.fc_enc(output)
        # enc = self.enc_ln(enc)
        noise = torch.normal(torch.zeros_like(enc), torch.ones_like(enc))
        enc = enc + noise
        # enc = self.enc_noise_ln(enc)

        gen = self.fc_out_init(enc)
        for layer in self.fc_res_out_layers:
            gen = layer(gen)
        gen = self.fc_out_final(gen)

        return gen


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         self.n_future = 10
#         self.bar_embedder = BarEmbedder()
#         self.pos_enc = nn.Parameter(torch.normal(torch.zeros(WINDOW + self.n_future, D_MODEL), torch.ones(WINDOW + self.n_future, D_MODEL)))
#         self.enc_ln = nn.LayerNorm(D_MODEL)
#
#         self.gen_encoder = GenEncoder()
#         self.gen_decoder = GenDecoder()
#
#     def forward(self, market_values):
#         """
#         market_values of size (batch_size, seq_len, D_BAR)
#         """
#         device = market_values.device
#         batch_size = market_values.size()[0]
#         input_seq_len = market_values.size()[1]
#
#         market_values = market_values[:, :, 3].view(batch_size, input_seq_len, 1)
#         means = market_values.contiguous().view(batch_size, -1).mean(1).view(-1, 1, 1)
#         stds = market_values.contiguous().view(batch_size, -1).std(1).view(-1, 1, 1)
#         market_values = (market_values - means) / (stds + 1e-9)
#
#         # market_values_shifted = torch.cat([market_values[:, 0, 0].view(batch_size, 1, 1), market_values[:, :-1, 3].view(batch_size, WINDOW - 1, 1)], dim=1)
#         # market_values_ = torch.cat([market_values_shifted, market_values], dim=2)
#
#         input = self.bar_embedder(market_values) + self.pos_enc[:WINDOW].repeat(batch_size, 1, 1)
#         input = self.enc_ln(input)
#         input = self.gen_encoder(input)
#
#         generated = []
#         gen_enc = torch.zeros(batch_size, 1, D_MODEL)
#         for i_gen in range(self.n_future):
#             gen = self.gen_decoder(input, gen_enc)
#             # print(gen[:, -1].mean(0))
#
#             if i_gen == 0:
#                 res = market_values[:, -1].view(batch_size, 1, 1)
#             else:
#                 res = generated[-1].view(batch_size, 1, 1)
#
#             cur_gen = gen[:, -1].view(batch_size, 1, 1) + res
#             generated.append(cur_gen.view(batch_size, 1, 1))
#
#             new_enc = (self.bar_embedder(cur_gen) + self.pos_enc[WINDOW + i_gen].repeat(batch_size, 1, 1))
#             new_enc = self.enc_ln(new_enc)
#             gen_enc = torch.cat([gen_enc, new_enc.contiguous()], dim=1)
#
#         generated = torch.cat(generated, dim=1) * (stds + 1e-9) + means
#         return generated


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.n_future = 10
        self.enc_dim = 16
        self.init_generator = nn.Sequential(
                                nn.Linear(WINDOW + 2 * self.n_future, D_MODEL),
                                nn.LayerNorm(D_MODEL),
                                nn.LeakyReLU()
                            )
        self.n_init_generator_res_layers = 2
        self.init_generator_res_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_init_generator_res_layers)])
        self.init_enc = nn.Linear(D_MODEL, self.enc_dim)

        # self.final_enc = nn.Sequential(
        #                         nn.Linear(self.enc_dim, D_MODEL),
        #                         nn.LayerNorm(D_MODEL),
        #                         nn.LeakyReLU()
        #                     )
        # self.n_final_generator_res_layers = 0
        # self.final_generator_res_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_final_generator_res_layers)])
        # self.final_generator = nn.Linear(D_MODEL, 1)

        self.init_to_enc = nn.Sequential(
                                self.init_generator,
                                *self.init_generator_res_layers,
                                self.init_enc,
                            )
        self.enc_to_mu = nn.Linear(self.enc_dim, self.enc_dim)
        self.enc_to_sigma = nn.Linear(self.enc_dim, self.enc_dim)

        # self.enc_ln = nn.LayerNorm(self.enc_dim)
        # self.enc_noise_ln = nn.LayerNorm(self.enc_dim)

        # self.enc_to_final = nn.Sequential(
        #                         self.final_enc,
        #                         *self.final_generator_res_layers,
        #                         self.final_generator,
        #                     )
        self.enc_to_final = nn.Sequential(
                                nn.Linear(self.enc_dim, D_MODEL),
                                nn.LeakyReLU(),
                                nn.Linear(D_MODEL, 1)
                            )

    def forward(self, market_values):
        """
        market_values of size (batch_size, seq_len, D_BAR)
        """
        device = market_values.device
        batch_size = market_values.size()[0]
        input_seq_len = market_values.size()[1]

        market_values = market_values[:, :, 3].view(batch_size, input_seq_len)
        means = market_values.contiguous().view(batch_size, -1).mean(1).view(-1, 1)
        stds = market_values.contiguous().view(batch_size, -1).std(1).view(-1, 1)
        market_values = (market_values - means) / (stds + 1e-9)
        market_values = market_values.view(batch_size, input_seq_len).contiguous()

        generated = [torch.zeros(batch_size, 1) for _ in range(self.n_future)]
        for i_gen in range(self.n_future):
            one_hot = torch.zeros(batch_size, self.n_future)
            one_hot[:, i_gen] = 1
            enc = self.init_to_enc(torch.cat([market_values] + generated + [one_hot], dim=1))

            mu = self.enc_to_mu(enc)
            sigma = torch.log(1 + torch.exp(self.enc_to_sigma(enc)))
            z = torch.normal(torch.zeros_like(mu), torch.ones_like(sigma))
            enc = (z * sigma) + mu

            gen = self.enc_to_final(enc)

            if i_gen == 0:
                res = market_values[:, -1].view(batch_size, 1)
            else:
                res = generated[i_gen-1].view(batch_size, 1)

            cur_gen = gen[:, -1].view(batch_size, 1) + res
            generated[i_gen] = cur_gen.view(batch_size, 1)

        generated = torch.cat(generated, dim=1) * (stds + 1e-9) + means
        generated = generated.view(batch_size, self.n_future, 1).contiguous()
        return generated


class DiscEncoder(nn.Module):
    def __init__(self):
        super(DiscEncoder, self).__init__()

        self.n_attn_layers = 4
        self.attn_layers = nn.ModuleList([EncoderLayer() for _ in range(self.n_attn_layers)])

    def forward(self, input):

        for layer in self.attn_layers:
            input = layer(input)

        return input


class DiscDecoder(nn.Module):
    def __init__(self):
        super(DiscDecoder, self).__init__()

        self.n_attn_layers = 4
        self.attn_layers = nn.ModuleList([DecoderLayer() for _ in range(self.n_attn_layers)])

    def forward(self, input, output):

        for layer in self.attn_layers:
            output = layer(input, output)

        return output


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.n_future = 10
#         self.bar_embedder = BarEmbedder()
#         self.pos_enc = nn.Parameter(torch.normal(torch.zeros(WINDOW + self.n_future, D_MODEL), torch.ones(WINDOW + self.n_future, D_MODEL)))
#         self.enc_ln = nn.LayerNorm(D_MODEL)
#
#         self.disc_encoder = DiscEncoder()
#         self.disc_decoder = DiscDecoder()
#
#         self.n_fc_res_layers = 1
#         self.fc_res_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_fc_res_layers)])
#         self.fc_disc = nn.Linear(D_MODEL, 1)
#
#         for param in [self.fc_disc]:
#             nn.init.normal_(param.weight, mean=0, std=0.02)
#             nn.init.normal_(param.bias, mean=0, std=0.02)
#
#     def forward(self, market_values, query):
#         """
#         market_values of size (batch_size, seq_len, D_BAR)
#         """
#         device = market_values.device
#         batch_size = market_values.size()[0]
#         seq_len = market_values.size()[1] + query.size()[1]
#
#         inputs = torch.cat([market_values, query], dim=1)
#         means = inputs.contiguous().view(batch_size, -1).mean(1).view(-1, 1, 1)
#         stds = inputs.contiguous().view(batch_size, -1).std(1).view(-1, 1, 1)
#         inputs = (inputs - means) / (stds + 1e-9)
#
#         # inputs_shifted = torch.cat([inputs[:, 0, 0].view(batch_size, 1, 1), inputs[:, :-1, 3].view(batch_size, WINDOW + self.n_future - 1, 1)], dim=1)
#         # inputs_ = torch.cat([inputs_shifted, inputs], dim=2)
#
#         inputs = self.bar_embedder(inputs) + self.pos_enc.repeat(batch_size, 1, 1)
#         inputs = self.enc_ln(inputs)
#         encoding = self.disc_encoder(inputs)
#
#         output = torch.zeros(batch_size, 1, D_MODEL)
#         disc = self.disc_decoder(encoding, output)
#         disc = disc[:, -1]
#
#         for layer in self.fc_res_layers:
#             disc = layer(disc)
#         disc = self.fc_disc(disc)
#         disc = torch.sigmoid(disc)
#
#         return disc


class ConditionedDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionedDiscriminator, self).__init__()

        self.n_future = 10
        self.init_fc = nn.Linear(WINDOW + self.n_future, D_MODEL)
        self.init_ln = nn.LayerNorm(D_MODEL)

        self.n_fc_res_layers = 2
        self.fc_res_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_fc_res_layers)])
        self.fc_disc = nn.Linear(D_MODEL, 1)

        for param in [self.init_fc, self.fc_disc]:
            nn.init.normal_(param.weight, mean=0, std=0.02)
            nn.init.normal_(param.bias, mean=0, std=0.02)

    def forward(self, market_values, query):
        """
        market_values of size (batch_size, seq_len)
        """
        device = market_values.device
        batch_size = market_values.size()[0]
        seq_len = market_values.size()[1] + query.size()[1]

        inputs = torch.cat([market_values, query], dim=1).view(batch_size, seq_len)
        means = inputs.contiguous().view(batch_size, -1).mean(1).view(-1, 1)
        stds = inputs.contiguous().view(batch_size, -1).std(1).view(-1, 1)
        inputs = (inputs - means) / (stds + 1e-9)

        disc = self.init_fc(inputs)
        disc = self.init_ln(disc)
        disc = F.leaky_relu(disc)

        for layer in self.fc_res_layers:
            disc = layer(disc)
        disc = self.fc_disc(disc)
        disc = torch.sigmoid(disc)

        return disc

class UnconditionedDiscriminator(nn.Module):
    def __init__(self):
        super(UnconditionedDiscriminator, self).__init__()

        self.n_future = 10
        self.init_fc = nn.Linear(self.n_future, D_MODEL)
        self.init_ln = nn.LayerNorm(D_MODEL)

        self.n_fc_res_layers = 2
        self.fc_res_layers = nn.ModuleList([FCResLayer() for _ in range(self.n_fc_res_layers)])
        self.fc_disc = nn.Linear(D_MODEL, 1)

        for param in [self.init_fc, self.fc_disc]:
            nn.init.normal_(param.weight, mean=0, std=0.02)
            nn.init.normal_(param.bias, mean=0, std=0.02)

    def forward(self, query):
        """
        query of size (batch_size, n_future)
        """
        device = query.device
        batch_size = query.size()[0]
        seq_len = query.size()[1]

        inputs = query.view(batch_size, seq_len)
        means = inputs.contiguous().view(batch_size, -1).mean(1).view(-1, 1)
        stds = inputs.contiguous().view(batch_size, -1).std(1).view(-1, 1)
        inputs = (inputs - means) / (stds + 1e-9)

        disc = self.init_fc(inputs)
        disc = self.init_ln(disc)
        disc = F.leaky_relu(disc)

        for layer in self.fc_res_layers:
            disc = layer(disc)

        disc = self.fc_disc(disc)
        disc = torch.sigmoid(disc)

        return disc


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.init_fc = nn.Linear(WINDOW, D_MODEL)
        self.init_ln = nn.LayerNorm(D_MODEL)

        self.n_res = 2
        self.res_layers = nn.Sequential(*nn.ModuleList([FCResLayer() for _ in range(self.n_res)]))
        self.final_fc = nn.Linear(D_MODEL, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, market_values):
        """
        market_values of size (batch_size, seq_len, D_BAR)
        """
        device = market_values.device
        batch_size = market_values.size()[0]
        seq_len = market_values.size()[1]

        inputs = market_values[:, :, 3].view(batch_size, seq_len)
        mean = inputs.contiguous().view(batch_size, -1).mean(1).view(-1, 1)
        std = inputs.contiguous().view(batch_size, -1).std(1).view(-1, 1)
        inputs = (inputs - mean) / (std + 1e-9)

        x = self.init_fc(inputs)
        x = self.init_ln(x)

        x = self.res_layers(x)

        x = self.final_fc(x)
        x = self.softmax(x)

        return x, inputs[:, -1].view(batch_size, 1), mean, std
