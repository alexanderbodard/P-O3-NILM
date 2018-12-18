"""
Functions to perform basic analysis of REDD data
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

from files import read_dat_file, read_npy_file


THRESHOLD    = 60                                                              # Time gap threshold for moving data into separate files
PATH         = "C:/users/arneb/P&O3/Data/REDD/data/low_freq"                   # Path to the low_freq folder of the REDD dataset


def analyse_time_gaps(house, channel, path=PATH, threshold=THRESHOLD):
    """
    Analyses the time gaps in the .dat file of the given house and channel
    """
    
    times, data = read_dat_file(path + f"/house_{house}/channel_{channel}.dat")
    
    diffs = np.diff(times)
    diff_counts = np.array(np.unique(diffs, return_counts=True)).transpose()
    
    plt.plot(diffs)
    plt.show()
    
    print("\n\n******************************\n\n")
    
    print("  Time gap: Number of occurences\n")
    for d, c in diff_counts:
        print(f"{d:10d}: {c:10d}")

    print("\n\n******************************\n\n")
    
    print("Time lost with current threshold: ", sum(diffs[abs(diffs) >= threshold]))


def skipped_time(times, threshold=THRESHOLD):
    
    diffs = np.diff(times)
    
    lt_threshold = np.array([diffs, np.arange(len(diffs))]).transpose()[diffs > threshold]
    
    skipped = [(0, times[0])] + [(times[i], times[i + 1]) for _, i in lt_threshold]
    return skipped


def time_interval(house, channel, path=PATH):
    """
    Prints the starting and end dates of the data at the given house and channel
    """
    
    times, _ = read_dat_file(path + f"/house_{house}/channel_{channel}.dat")
    
    print(datetime.datetime.utcfromtimestamp(times[0]).strftime('%d-%m-%Y %H:%M:%S')
          + "\nto\n" + 
          datetime.datetime.utcfromtimestamp(times[-1]).strftime('%d-%m-%Y %H:%M:%S'))
    
    return times[0], times[-1]
