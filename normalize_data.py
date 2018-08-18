import time
import pandas as pd


def time_to_float(time_, normalize=True):
    if normalize:
        return (time_.tm_hour * 60 + time_.tm_min) / (60 * 24)

    return time_.tm_hour * 60 + time_.tm_min


def normalize_and_save(source):
    t0 = time.time()
    print("normalizing market values")
    data = pd.DataFrame(pd.read_csv(source, header=None)).loc[:, 1:5]
    normalization_factor = data.loc[:, 2:].mean().mean()    # normalize with the mean of all values
    print("normalization factor: {0}".format(normalization_factor))
    data.loc[:, 2:] /= normalization_factor
    data.columns = ['time', 'open', 'high', 'low', 'close']
    print("finished normalizing market values")

    print()

    print("normalizing time column")
    time_col = pd.Series()
    for i, time_ in enumerate(data['time']):
        if i % int(data.shape[0] / 100) == 0:
            print("{0}% of rows completed".format(round(i*100 / data.shape[0])))
        time_float = time.strptime(time_, "%H:%M")
        time_col = time_col.append(pd.Series([time_to_float(time_float)]), ignore_index=True)

    data['time'] = time_col
    print("finished normalizing time column")

    print()

    print("saving")
    data.to_csv("normalized_data/" + source.split("/")[1][:-4] + "-" + str(normalization_factor) + ".csv")
    print("normalization completed in {0} seconds".format(time.time() - t0))

    print()


normalize_and_save("data/DAT_MT_EURUSD_M1_2017.csv")
