import json
import numpy as np
from sklearn.preprocessing import normalize
import random

def load_bars(file):
    '''
        Do we normalize everything together:
            - Open, High, Low, Close, Volume, <Indicator>
        
        Or do we normalize the groups:
            - Open, High, Low, Close
            - Volume,
            - <Indicator>

        Or do we normalize everything independently:
            - Open
            - High
            - Low
            - Close
            - Volume
            - <Indicator>
    '''
    bars = json.load(open("res/series/"+file))['bars']
    raw_bars = [[bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']] for bar in bars]
    # volumes = [bar['volume'] for bar in bars]
    
    normalized_bars = [list(bar) for bar in normalize(raw_bars, axis=0)] # Normalize independently
    # normalized_volumes = normalize(np.reshape(volumes, (-1, 1)), axis=0)
    
    # for index in range(len(bars)):
    #     normalized_bars[index].append(normalized_volumes[index][0])
    #     raw_bars[index].append(volumes[index])

    return raw_bars, normalized_bars


def load_actions_bars(file, sample_size, reward_after, min_pip):
    raw_bars, bars = load_bars(file)

    data = []
    actions = []
    for i in range(sample_size, len(bars)+1):
        sample = []
        try:
            # Calculate pips
            current_bar = raw_bars[i-1]
            reward_bar = raw_bars[i+reward_after-1]
            buy_pips = calc_pips(current_bar[-2], reward_bar[-2], 0)
            sell_pips = calc_pips(current_bar[-2], reward_bar[-2], 1)

            # Assign action label
            action_matrix = [0 for i in range(3)]
            action = 2
            if buy_pips > min_pip or sell_pips > min_pip:
                action = np.argmax([buy_pips, sell_pips])
            action_matrix[action] = 1
            actions.append(np.array(action_matrix))

            # load batch
            for bar in bars[i-sample_size:i]:
                sample.append(np.array(bar))
            data.append(np.array(sample))
        except:
            pass
    return (np.array(data), np.array(actions))


def load_labeled_bars(file, sample_size):
    raw_bars, bars = load_bars(file)
    labels = json.load(open("res/labels/"+file))

    data = []
    actions = []
    for i in range(sample_size, len(bars)+1):
        sample = []
        try:
            # Assign action label
            action = str_to_action(labels[i-1])
            action_matrix = [0 for i in range(3)]
            action_matrix[action] = 1
            actions.append(np.array(action_matrix))

            # load batch
            for bar in bars[i-sample_size:i]:
                sample.append(np.array(bar))
            data.append(np.array(sample))
        except:
            pass

    return (np.array(data), np.array(actions))


def load_states(file, sample_size, ema=None, rsi=False, max_bars=None):
    bars, _ = load_bars(file)
    len_bars = len(bars)
    if max_bars is not None and max_bars < len_bars:
        bars = bars[len_bars-max_bars:]

    if rsi:
        bars = add_rsi(bars)
    if ema is not None:
        bars = add_ema(bars, ema)


    data = [] # Non normalized
    data_bar = []
    norm = [] # Normalized
    norm_bar = []
    last = None
    for i in range(sample_size, len(bars)+1):
        sample = []
        try:
            # load batch
            for bar in bars[i-sample_size:i]:
                sample.append(np.array(bar))
            # TODO Do we need to keep volume in here?
            sample = np.array(sample)
            data_bar.append(sample[-1])
            sample_norm = normalize(sample, axis=0)
            norm_bar.append(sample_norm[-1])

            if last is not None:
                data.append((last[0], sample))
                norm.append((last[1], sample_norm))
            
            last = (sample, sample_norm)
        except:
            pass

    return (np.array(data), np.array(data_bar), np.array(norm), np.array(norm_bar))


def add_rsi(bars, period=14):
    close_prices = [bar[-2] for bar in bars]
    prev_avg_gain = 0.0
    prev_avg_loss = 0.0
    return_bars = []
    for index in range(period+1, len(close_prices)+1):
        sample = close_prices[index-period-1:index]
        changes = []
        for i in range(len(sample)-1, 0, -1):
            changes.insert(0, sample[i] - sample[i-1])
        change = changes[-1]

        gain = change if change > 0.0 else 0.0
        loss = abs(change) if change <= 0.0 else 0.0
        avg_gain = sum([c for c in changes if c > 0.0]) / period
        avg_loss = sum([abs(c) for c in changes if c <= 0.0]) / period
        prev_avg_gain = avg_gain
        prev_avg_loss = avg_loss

        if index == period+1:
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
        else:
            rs = (((prev_avg_gain * (period-1)) + gain) / period) / (((prev_avg_loss * (period-1)) + loss) / period)
        bar = bars[index-1][:]
        bar.insert(0, 100.0 - (100.0 / (1.0 + rs)))
        return_bars.append(bar)

    # TODO Should we return everything after the first 250 in order to get a more accurate RSI?
    # TODO Should we return a score instead of the actual values??
    return return_bars[250:]


def add_sma(bars, period=20):
    close_prices = [bar[-2] for bar in bars]
    return_bars = []
    for index in range(period, len(close_prices) + 1):
        sample = close_prices[index - period : index]
        avg = sum(sample) / float(period)
        bar = bars[index-1][:]
        bar.insert(0, avg)
        return_bars.append(bar)
    return return_bars


def add_ema(bars, periods=[20]):
    emas = []
    return_bars = [bar[:] for bar in bars]
    for period in periods:
        multiplier = (2.0 / (float(period) + 1.0))
        sma_bars = add_sma(bars, period)
        ema = []
        for index in range(len(sma_bars)):
            bar = sma_bars[index]
            avg = bar[0]
            close_price = bar[-2]
            if index == 0:
                ema.append(avg)
            else:
                last_ema = sma_bars[index-1][0]
                ema.append(((close_price - last_ema) * multiplier) + last_ema)
        emas.append(ema)
    
    fewest = min([len(ema) for ema in emas])
    for ema in emas:
        while len(ema) > fewest:
            ema.pop(0)
    while len(return_bars) > fewest:
        return_bars.pop(0)
    
    for index in range(len(return_bars)):
        bar = return_bars[index]
        for ema in emas:
            bar.insert(0, ema[index])
    
    return return_bars


def str_to_action(value):
    value = value.lower()
    if value == "wait":
        return 2
    elif value == "sell":
        return 1
    elif value == "buy":
        return 0
    else:
        raise "Invalid string"


def action_to_str(action):
    if action == 2:
        return "wait"
    elif action == 1:
        return "sell"
    elif action == 0:
        return "buy"
    else:
        raise "Invalid action"


def calc_pips(start, end, action):
    # This only works for non JPY symbols
    return (start - end if action == 1 else end - start) * 10000.0
