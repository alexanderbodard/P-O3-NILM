"""
Functions to preprocess data to prepare it for training
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from files       import read_file, write_file, get_max_power, read_npy_file
from interpolate import interpolate_channel, remove_negative_diffs, join_results
from analyse     import skipped_time


THRESHOLD    = 60                                                              # Time gap threshold for moving data into separate files
PATH         = "C:/users/arneb/P&O3/Data/REDD/data/low_freq"                   # Path to the low_freq folder of the REDD dataset
HOUSES       = [1,2,3,4,5,6]
CHANNELS     = {house: list(range(3, [20, 11, 22, 20, 26, 17][house - 1] + 1)) for house in HOUSES}
INPUT_FTYPE  = ("dat", "npy")[0]                                               # File type to load the data from or store it in; 
OUTPUT_FTYPE = ("dat", "npy")[1]                                               #  either "dat" for the default REDD data format in a .dat
                                                                               #  file or "npy" for a numpy .npy file holding the array,
                                                                               #  allowing for fast saving to and loading from files

def match_times_channel(house_or_times, channel_or_data, path=PATH, threshold=THRESHOLD, input_ftype=INPUT_FTYPE, output_ftype=OUTPUT_FTYPE, mean=False, scale=1):
    """
    Preprocesses the data in the given channel of the given house, saves the parts
    of the interpolated data where there is data present from both the mains channel
    and the actual channel
    """
    
    if output_ftype == "dat": raise ValueError
    
    if input_ftype:
        house, channel    = house_or_times, channel_or_data
        times_c, data_c   = read_file(path + f"/house_{house}/channel_{channel}.{input_ftype}", input_ftype)
        times_m, data_m_1 = read_file(path + f"/house_{house}/channel_1.{input_ftype}", input_ftype)
        _      , data_m_2 = read_file(path + f"/house_{house}/channel_2.{input_ftype}", input_ftype)
        
        data_m            = data_m_1 + data_m_2
        
        times_c, data_c = remove_negative_diffs(times_c, data_c, input_ftype=None, output_ftype=None)
        times_m, data_m = remove_negative_diffs(times_m, data_m, input_ftype=None, output_ftype=None)
    
        times_c, data_c = join_results(interpolate_channel(times_c, data_c, threshold=threshold, input_ftype=None, output_ftype=None))
        times_m, data_m = join_results(interpolate_channel(times_m, data_m, threshold=threshold, input_ftype=None, output_ftype=None))
        
    else:
        times_c, times_m, *house = house_or_times
        data_c, data_m, *channel = channel_or_data
        
        if not house:
            house = "undefined"
        else:
            house = house[0]
            
        if not channel:
            channel = "undefined"
        else:
            channel = channel[0]
        
    skipped_c = skipped_time(times_c, threshold=THRESHOLD)
    skipped_m = skipped_time(times_m, threshold=THRESHOLD)

    begin_times = [times_c[0], times_m[0]]
    end_times   = [times_c[-1], times_m[-1]]
    
    skipped = sorted(skipped_c + skipped_m)
    i = 0
    while i < len(skipped) - 1:
        (b1, e1), (b2, e2) = skipped[i : i + 2]
        if b2 < e1:
            # Overlap
            skipped[i: i + 2] = [(b1, max(e1, e2))]
        else:
            i += 1

    l       = []
    ind_c_e = np.argmax(times_c >= max(begin_times))
    ind_m_e = np.argmax(times_m >= max(begin_times))
    for b, e in skipped:

        ind_c_b = np.argmax(times_c > b)
        ind_m_b = np.argmax(times_m > b)

        times_c_slice = times_c[ind_c_e : ind_c_b]
        data_c_slice  = data_c [ind_c_e : ind_c_b]
        times_m_slice = times_m[ind_m_e : ind_m_b]
        data_m_slice  = data_m [ind_m_e : ind_m_b]

        if times_c_slice.size and times_m_slice.size:
            l.append((np.copy(times_c_slice), np.copy(data_c_slice) / scale, 
                      np.copy(times_m_slice), (np.copy(data_m_slice) - np.mean(data_m_slice) * mean) / scale))
        
        ind_c_e = np.argmax(times_c >= e)
        ind_m_e = np.argmax(times_m >= e)
        
    c_min_end = np.argmax(times_c >= min(end_times))
    m_min_end = np.argmax(times_m >= min(end_times))
    times_c_slice = times_c[ind_c_e : c_min_end + 1]
    data_c_slice  = data_c [ind_c_e : c_min_end + 1]
    times_m_slice = times_m[ind_m_e : m_min_end + 1]
    data_m_slice  = data_m [ind_m_e : m_min_end + 1]
    
    l.append((np.copy(times_c_slice), np.copy(data_c_slice) / scale, 
              np.copy(times_m_slice), (np.copy(data_m_slice) - np.mean(data_m_slice) * mean) / scale))
    
    if output_ftype:
        for file_num, (tcs, dcs, tms, dms) in enumerate(l):
            if not os.path.exists(path + f"/house_{house}/channel_{channel}_preprocessed"):
                os.mkdir(path + f"/house_{house}/channel_{channel}_preprocessed")
            
            data_array = np.append(dcs, dms)

            write_file(path + f"/house_{house}/channel_{channel}_preprocessed/file_{file_num}.npy", tcs, data_array, output_ftype)
    
    else:
        return l
        
    
def preprocess(houses=HOUSES, channels=CHANNELS, path=PATH, threshold=THRESHOLD, input_ftype=INPUT_FTYPE, output_ftype=OUTPUT_FTYPE):
    """
    Maps the match_times_channel function to all of the given channels
    """
    
    ls = []
    for house in houses:
        times_m, data_m_1 = read_file(path + f"/house_{house}/channel_1.{input_ftype}", input_ftype)
        _      , data_m_2 = read_file(path + f"/house_{house}/channel_2.{input_ftype}", input_ftype)
        
        data_m = data_m_1 + data_m_2
        
        times_m, data_m = remove_negative_diffs(times_m, data_m, input_ftype=None, output_ftype=None)
        times_m, data_m = join_results(interpolate_channel(times_m, data_m, threshold=threshold, input_ftype=None, output_ftype=None))
        
        for channel in channels[house]:
            times_c, data_c = read_file(path + f"/house_{house}/channel_{channel}.{input_ftype}", input_ftype)
            times_c, data_c = remove_negative_diffs(times_c, data_c, input_ftype=None, output_ftype=None)
            times_c, data_c = join_results(interpolate_channel(times_c, data_c, threshold=threshold, input_ftype=None, output_ftype=None))
            
            max_power = get_max_power(house, channel, path)
            times_m_copy, data_m_copy = np.copy(times_m), np.copy(data_m)
            
            l = match_times_channel((times_c, times_m_copy, house), (data_c, data_m_copy, channel),
                                    path=path, threshold=threshold, input_ftype=None,
                                    output_ftype=output_ftype, scale=max_power, mean=True)
            ls.append(l)
            
    if not output_ftype: return ls
    
    

def plot_channel(house, channel, path=PATH):
    preprocess([house], {house: [channel]})
    for i in range(1000):
        try:
            t,d = read_npy_file(path + f"/house_{house}/channel_{channel}_preprocessed/file_{i}.npy")
        except:
            break
        
        print(i)
        d1, d2 = np.split(d, 2)
        
        plt.plot(d1)
        plt.show()
    
    