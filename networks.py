import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import time

torch.manual_seed(0)
D_BAR = 5
D_MODEL = 256
WINDOW = 180
P_DROPOUT = 0.25
# torch.cuda.manual_seed(0)
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

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
        self.conv1_bn = nn.BatchNorm1d(self.n_channels[0], eps=1e-9)

        self.conv2 = nn.Conv1d(in_channels=self.n_channels[0],
                                out_channels=self.n_channels[1],
                                kernel_size=self.kernel_sizes[1])
        self.conv2_bn = nn.BatchNorm1d(self.n_channels[1], eps=1e-9)

        self.conv3 = nn.Conv1d(in_channels=self.n_channels[1],
                                out_channels=self.n_channels[2],
                                kernel_size=self.kernel_sizes[2])
        self.conv3_bn = nn.BatchNorm1d(self.n_channels[2], eps=1e-9)

        self.conv4 = nn.Conv1d(in_channels=self.n_channels[2],
                                out_channels=self.n_channels[3],
                                kernel_size=self.kernel_sizes[3])
        self.conv4_bn = nn.BatchNorm1d(self.n_channels[3], eps=1e-9)

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
        time_states = []
        for time_state in market_values:
            time_states.append(time_state.view(-1, D_BAR, 1))

        time_states = torch.cat(time_states, dim=2)

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

        x = x.squeeze().view(-1, self.L_in * self.n_channels[-1])

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
        self.conv1_bn = nn.BatchNorm1d(self.n_channels[0], eps=1e-9)

        self.conv2 = nn.Conv1d(in_channels=self.n_channels[0],
                                out_channels=self.n_channels[1],
                                kernel_size=self.kernel_sizes[1])
        self.conv2_bn = nn.BatchNorm1d(self.n_channels[1], eps=1e-9)

        self.pool = nn.AvgPool1d(self.pool_kernel_size)


        self.L_in = WINDOW
        for k in self.kernel_sizes:
            self.L_in = math.floor(self.L_in - k + 1)
            self.L_in = math.floor((self.L_in - self.pool_kernel_size) / self.pool_kernel_size + 1)
        self.L_in = int(self.L_in)

        self.n_fc_layers = 1
        self.conv_out_dim = self.L_in * self.n_channels[-1]
        begin_dim = self.conv_out_dim
        end_dim = D_MODEL
        # scale down the layer size linearly from begin_dim to end_dim
        self.fc_layers = nn.ModuleList([nn.Linear(int((end_dim - begin_dim) * n / self.n_fc_layers + begin_dim), int((end_dim - begin_dim) * (n + 1) / self.n_fc_layers + begin_dim)) for n in range(self.n_fc_layers)])
        self.fc_bn = nn.ModuleList([nn.BatchNorm1d(int((end_dim - begin_dim) * (n + 1) / self.n_fc_layers + begin_dim), eps=1e-9) for n in range(self.n_fc_layers)])

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


class MarketEncoder(nn.Module):
    """
    DEPRECATED
    """

    def __init__(self, device='cuda'):
        super(MarketEncoder, self).__init__()

        self.n_lstm_layers = 2

        self.fc1 = nn.Linear(D_BAR, D_MODEL)
        self.lstm = nn.LSTM(input_size=D_MODEL, hidden_size=D_MODEL, num_layers=self.n_lstm_layers)
        self.hidden = self.init_hidden(1, device)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.n_lstm_layers, batch_size, D_MODEL, device=device),
                torch.zeros(self.n_lstm_layers, batch_size, D_MODEL, device=device))

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

    def __init__(self):
        super(AttentionMarketEncoder, self).__init__()

        self.Ns = [4]
        self.h = 8
        self.fc_out_middle_size = D_MODEL

        self.d_k = int(D_MODEL / self.h)
        self.d_v = int(D_MODEL / self.h)

        self.n_entities = WINDOW

        self.fc_bar = nn.Linear(D_BAR + 1, D_MODEL)

        self.WQs = nn.ModuleList([nn.Linear(D_MODEL, self.d_k, bias=False) for _ in range(self.h * len(self.Ns))])
        self.WKs = nn.ModuleList([nn.Linear(D_MODEL, self.d_k, bias=False) for _ in range(self.h * len(self.Ns))])
        self.WVs = nn.ModuleList([nn.Linear(D_MODEL, self.d_v, bias=False) for _ in range(self.h * len(self.Ns))])
        self.WOs = nn.ModuleList(nn.Linear(self.h * self.d_v, D_MODEL, bias=False) for _ in range(len(self.Ns)))

        self.a_gain_ = nn.ParameterList([nn.Parameter(torch.zeros(D_MODEL)) for _ in range(np.sum(self.Ns))])
        self.a_bias = nn.ParameterList([nn.Parameter(torch.zeros(D_MODEL)) for _ in range(np.sum(self.Ns))])

        self.fc_out1 = nn.ModuleList([nn.Linear(D_MODEL, self.fc_out_middle_size) for _ in range(np.sum(self.Ns))])
        self.fc_out2 = nn.ModuleList([nn.Linear(self.fc_out_middle_size, D_MODEL) for _ in range(np.sum(self.Ns))])
        self.fc_out_gain_ = nn.ParameterList([nn.Parameter(torch.zeros(D_MODEL)) for _ in range(np.sum(self.Ns))])
        self.fc_out_bias = nn.ParameterList([nn.Parameter(torch.zeros(D_MODEL)) for _ in range(np.sum(self.Ns))])

        self.fc_final = nn.Linear(WINDOW, 1)

    def forward(self, market_values):
        time_states = []
        for i, time_state in enumerate(market_values):
            time_tag = torch.Tensor([(i - WINDOW / 2) / (WINDOW / 2)], device=time_state.device).repeat(time_state.size()[0]).view(-1, 1)
            # time_tag.device = time_state.device
            if time_state.device == torch.device('cuda:' + str(torch.cuda.current_device())):
                time_tag = time_tag.cuda()

            time_state_ = torch.cat([time_state, time_tag], dim=1).view(1, -1, D_BAR+1)
            time_state_ = self.fc_bar(time_state_)
            time_states.append(time_state_)

        inputs = torch.cat(time_states, dim=0)

        for i_N, N in enumerate(self.Ns):
            n_N = int(np.sum(self.Ns[:i_N]))
            for j in range(N):
                heads = []
                for i in range(self.h):
                    Q = self.WQs[i_N*self.h + i](inputs)
                    K = self.WKs[i_N*self.h + i](inputs)
                    V = self.WVs[i_N*self.h + i](inputs)
                    # print(Q, K, V)

                    saliencies = torch.bmm(Q, K.transpose(1, 2))
                    weights = F.softmax(saliencies / math.sqrt(self.d_k), dim=2)
                    # print(N, j, weights.max(dim=2)[0].mean())
                    head = torch.bmm(weights, V)
                    heads.append(head)

                heads = torch.cat(heads, dim=2)
                out = self.WOs[i_N](heads) + inputs
                out_mean = out.mean(dim=2).view(WINDOW, -1, 1)
                out_std = out.std(dim=2).view(WINDOW, -1, 1)
                out = (out - out_mean) / (out_std + 1e-9)
                out = out * (self.a_gain_[n_N + j] + 1) + self.a_bias[n_N + j]

                out = F.leaky_relu(self.fc_out1[n_N + j](out))
                out = self.fc_out2[n_N + j](out) + inputs
                out_mean = out.mean(dim=2).view(WINDOW, -1, 1)
                out_std = out.std(dim=2).view(WINDOW, -1, 1)
                inputs = (out - out_mean) / (out_std + 1e-9)
                inputs = inputs * (self.fc_out_gain_[n_N + j] + 1) + self.fc_out_bias[n_N + j]

        outputs = F.leaky_relu(self.fc_final(inputs.transpose(0, 2))).view(-1, D_MODEL)
        return outputs

    def layer_norm(self, layer, gain, bias):

        pass

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.initial_layer = nn.Linear(D_MODEL + 2, D_MODEL)
        self.initial_bn = nn.BatchNorm1d(D_MODEL, eps=1e-9)

        self.n_layers = 2
        self.layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL, eps=1e-9) for _ in range(self.n_layers)])

        self.final_layer = nn.Linear(D_MODEL, 2)

        self.dropout = nn.Dropout(P_DROPOUT)

    def forward(self, encoding, spread, log_steps):
        x = torch.cat([encoding.view(-1, D_MODEL), spread.view(-1, 1), log_steps.view(-1, 1)], 1)

        x = self.initial_layer(x)
        x = self.initial_bn(x)
        x = F.leaky_relu(x)
        for i in range(self.n_layers):
            if i % 2 == 0:
                if i == 0:
                    res = encoding.view(-1, D_MODEL)
                else:
                    res = x
                x = self.dropout(x)
                x = self.layers[i](x)
                x = self.bns[i](x)
                x = self.dropout(x)
                x = F.leaky_relu(x)
            else:
                x = self.layers[i](x)
                x = self.bns[i](x)
                x = x + res
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
        self.dist_bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL, eps=1e-9) for _ in range(self.dist_n_layers)])
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


class ActorCritic(nn.Module):
    """
    takes a market encoding (which also includes the percentage of balance
    already in a trade) and a proposed action, and outputs the policy (buy,
    sell, keep), and the value of the current state
    """

    def __init__(self):
        super(ActorCritic, self).__init__()

        self.d_action = 10
        self.d_action = 8
        self.dropout = nn.Dropout(P_DROPOUT)

        self.combined_initial = nn.Linear(D_MODEL, D_MODEL)
        self.combined_initial_bn = nn.BatchNorm1d(D_MODEL, eps=1e-9)
        self.n_combined_layers = 2
        self.combined_layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_combined_layers)])
        self.combined_bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL, eps=1e-9) for _ in range(self.n_combined_layers)])

        self.n_actor_layers = 2
        self.actor_layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_actor_layers)])
        self.actor_bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL, eps=1e-9) for _ in range(self.n_actor_layers)])
        self.actor_out = nn.Linear(D_MODEL, self.d_action)
        self.actor_softmax = nn.Softmax(dim=1)

        self.n_critic_layers = 2
        self.critic_layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_critic_layers)])
        self.critic_bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL, eps=1e-9) for _ in range(self.n_critic_layers)])
        self.critic_out = nn.Linear(D_MODEL, 1)

    def forward(self, market_encoding, temp=1):

        x = market_encoding.view(-1, D_MODEL)
        x = self.combined_initial(x)
        x = self.combined_initial_bn(x)
        x = F.leaky_relu(x)
        for i in range(self.n_combined_layers):
            if i % 2 == 0:
                if i == 0:
                    res = market_encoding.view(-1, D_MODEL)
                else:
                    res = x
                x = self.dropout(x)
                x = self.combined_layers[i](x)
                x = self.combined_bns[i](x)
                x = self.dropout(x)
                x = F.leaky_relu(x)
            else:
                x = self.combined_layers[i](x)
                x = self.combined_bns[i](x)
                x = x + res
                x = F.leaky_relu(x)

        policy = x
        for i in range(self.n_actor_layers):
            if i % 2 == 0:
                res = policy
                policy = self.dropout(policy)
                policy = self.actor_layers[i](policy)
                policy = self.actor_bns[i](policy)
                policy = self.dropout(policy)
                policy = F.leaky_relu(policy)
            else:
                policy = self.actor_layers[i](policy)
                policy = self.actor_bns[i](policy)
                policy = policy + res
                policy = F.leaky_relu(policy)

        policy_ = self.actor_out(policy) / temp
        policy = self.actor_softmax(policy_)

        critic = x
        for i in range(self.n_critic_layers):
            if i % 2 == 0:
                res = critic
                critic = self.dropout(critic)
                critic = self.critic_layers[i](critic)
                critic = self.critic_bns[i](critic)
                critic = self.dropout(critic)
                critic = F.leaky_relu(critic)
            else:
                critic = self.critic_layers[i](critic)
                critic = self.critic_bns[i](critic)
                critic = critic + res
                critic = F.leaky_relu(critic)

        critic = self.critic_out(critic)

        return policy, critic


class EncoderToOthers(nn.Module):

    def __init__(self):
        super(EncoderToOthers, self).__init__()
        self.initial_layer = nn.Linear(D_MODEL + 2, D_MODEL)
        self.initial_bn = nn.BatchNorm1d(D_MODEL, eps=1e-9)

        self.n_layers = 2
        self.layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(D_MODEL, eps=1e-9) for _ in range(self.n_layers)])
        self.dropout = nn.Dropout(P_DROPOUT)

    def forward(self, encoding, spread, percent_in):
        x = torch.cat([
                    encoding.view(-1, D_MODEL),
                    spread.view(-1, 1) * 10,
                    percent_in.view(-1, 1) * 10
                    ], 1)

        x = self.initial_layer(x)
        x = self.initial_bn(x)
        x = F.leaky_relu(x)
        for i in range(self.n_layers):
            if i % 2 == 0:
                if i == 0:
                    res = encoding.view(-1, D_MODEL)
                else:
                    res = x
                x = self.dropout(x)
                x = self.layers[i](x)
                x = self.bns[i](x)
                x = self.dropout(x)
                x = F.leaky_relu(x)
            else:
                x = self.layers[i](x)
                x = self.bns[i](x)
                x = x + res
                x = F.leaky_relu(x)

        return x
