import networks
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from zeus.zeus import Zeus

MEN = networks.CNNEncoder().cpu()
ETO = networks.EncoderToOthers().cpu()
PN = networks.ProbabilisticProposer().cpu()
MEN.load_state_dict(torch.load('./models/market_encoder.pt'))
ETO.load_state_dict(torch.load('./models/encoder_to_others.pt'))
PN.load_state_dict(torch.load('./models/proposer.pt'))
MEN = MEN.cpu()
ETO = ETO.cpu()
PN = PN.cpu()

instrument = np.random.choice(["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"])
start = np.random.randint(1136073600, 1546300800)
# instrument = "EUR_USD"
# start = np.random.randint(1546214400, 1546300800)

zeus = Zeus(instrument, "M1")

time_states = []
def add_bar(bar):
    time_state = [[[bar.open, bar.high, bar.low, bar.close, np.log(bar.volume + 1e-1)]]]

    if len(time_states) == 0 or time_state != time_states[-1]:
        time_states.append(time_state)
    else:
        return

    if len(time_states) >= networks.WINDOW:
        percent_in = zeus.position_size() / (abs(zeus.position_size()) + zeus.units_available() + 1e-9)

        input_time_states = torch.Tensor(time_states[-networks.WINDOW:]).view(networks.WINDOW, 1, networks.D_BAR).cpu()
        mean = input_time_states[:, 0, :4].mean()
        std = input_time_states[:, 0, :4].std()
        input_time_states[:, 0, :4] = (input_time_states[:, 0, :4] - mean) / std
        spread_normalized = bar.spread / std

        market_encoding = MEN.forward(input_time_states)
        market_encoding = ETO.forward(market_encoding, (std + 1e-9).log(), torch.Tensor([spread_normalized]), torch.Tensor([percent_in]))
        proposed, p_actions, w, mu, sigma = PN.forward(market_encoding, True)
        print("proposed:", proposed)
        print("probabilities:", p_actions)
        print()
        x = np.arange(0.001, 1, 0.001)
        y = np.array([PN.p(torch.Tensor([x_, x_]), w, mu, sigma).detach().numpy().squeeze() for x_ in x])
        plt.plot(x, y)
        plt.show()


zeus.stream_range(start, start + 60 * (networks.WINDOW + 1000), add_bar)
