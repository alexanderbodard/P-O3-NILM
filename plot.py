import numpy as np
import matplotlib.pyplot as plt
import datetime
import files
import rectangles
import predict

THRESHOLD = 60  # Time gap threshold for moving data into separate files
PATH = "P&O3Data/data/low_freq"


def get_labels(house, path=PATH):
    d = {}

    with open(path + f"/house_{house}/labels.dat", 'r') as file:
        for line in file.readlines():
            if not line: continue

            d.update({line.split()[0]: line.split()[1]})

    return d


def plot_graph(house, channel, end_time=100, begin_time=0, path=PATH):
    times, data = files.read_dat_file(path + f"/house_{house}/channel_{channel}.dat")

    plt.plot(times[begin_time:end_time], data[begin_time:end_time])
    plt.ylabel(get_labels(1)[str(channel)])
    plt.xlabel("time")
    plt.show()



