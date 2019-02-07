import networks
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from zeus.zeus import Zeus

MEN = networks.LSTMCNNEncoder().cpu()
ETO = networks.EncoderToOthers().cpu()
PN = networks.ProbabilisticProposer().cpu()
ACN = networks.ActorCritic().cpu()
MEN.load_state_dict(torch.load('./models/market_encoder.pt'))
ETO.load_state_dict(torch.load('./models/encoder_to_others.pt'))
PN.load_state_dict(torch.load('./models/proposer.pt'))
ACN.load_state_dict(torch.load('./models/actor_critic.pt'))
MEN = MEN.cpu()
ETO = ETO.cpu()
PN = PN.cpu()
ACN = ACN.cpu()

n_param = 0
for net in [MEN, ETO, PN, ACN]:
    for param in net.parameters():
        prod = 1
        for size_ in param.size():
            prod *= size_ if size_ != 0 else 1
        n_param += prod
print("n params all networks", n_param)

MEN.eval()
ETO.eval()
PN.eval()
ACN.eval()

instrument = np.random.choice(["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"])
start = np.random.randint(1136073600, 1546300800)
# instrument = "EUR_USD"
# start = np.random.randint(1546214400, 1546819200)

zeus = Zeus(instrument, "M1")

time_states = []
steps_since_last = 0
def add_bar(bar):
    global steps_since_last
    steps_since_last += 1
    time_state = [[[bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]]]

    if len(time_states) == 0 or time_state != time_states[-1]:
        time_states.append(time_state)
    else:
        return

    if len(time_states) >= networks.WINDOW and steps_since_last >= networks.WINDOW:
        percent_in = zeus.position_size() / (abs(zeus.position_size()) + zeus.units_available() + 1e-9)

        input_time_states = torch.Tensor(time_states[-networks.WINDOW:]).view(networks.WINDOW, 1, networks.D_BAR).cpu()
        mean = input_time_states[:, 0, :4].mean()
        std = input_time_states[:, 0, :4].std()
        input_time_states[:, 0, :4] = (input_time_states[:, 0, :4] - mean) / std
        spread_normalized = bar.spread / std

        market_encoding = MEN.forward(input_time_states)
        # for x in np.arange(-1, 1, 0.01):
        #     market_encoding_ = ETO.forward(market_encoding, torch.Tensor([spread_normalized]), torch.Tensor([x]))
        #     # proposed, p_actions, w, mu, sigma = PN.forward(market_encoding, True)
        #     proposed = torch.Tensor([[0.5, 0.5]])
        #     print(round(x, 2), ACN.forward(market_encoding_, proposed))
        market_encoding = ETO.forward(market_encoding, torch.Tensor([spread_normalized]), torch.Tensor([percent_in]))

        # temps = {'w':20, 'mu':2, 'sigma':2}
        temps = {'w':1, 'mu':1, 'sigma':1}
        proposed, p_actions, w, mu, sigma = PN.forward(market_encoding, True, temps)

        print("proposed:", proposed)
        print("probabilities:", p_actions)
        print("w:", w)
        print("mu:", mu)
        print("sigma:", sigma)
        print("ACN:", ACN.forward(market_encoding, proposed))
        print()
        x = np.arange(0.001, 1, 0.001)
        y = np.array([PN.p(torch.Tensor([x_, x_]), w, mu, sigma).detach().numpy().squeeze() for x_ in x])
        plt.plot(x, y)
        plt.show()



        steps_since_last = 0

n = 10000
while n > 0:
    n_ = min(500, n)
    zeus.stream_range(start, start + 60 * n_, add_bar)
    n -= n_
