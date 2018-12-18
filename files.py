"""
Functions to interact with REDD's .dat files and their structure,
as well as our custom .npy files
"""

import numpy as np
import os
import shutil


PATH       = "C:/users/arneb/P&O3/Data/REDD/data/low_freq"                     # Path to the low_freq folder of the REDD dataset
POWER_DICT = {
        "oven": 100,
        "refrigerator": 800,
        "dishwaser": 100,
        "kitchen_outlets": 100,
        "lighting": 100,
        "washer_dryer": 100,
        "microwave": 1000,
        "bathroom_gfi": 100,
        "electric_heat": 100,
        "stove": 100
        }

WIDTH_DICT = {
        "oven": 100,
        "refrigerator": 100,
        "dishwaser": 100,
        "kitchen_outlets": 100,
        "lighting": 100,
        "washer_dryer": 100,
        "microwave": 100,
        "bathroom_gfi": 100,
        "electric_heat": 100,
        "stove": 100
        }


def get_labels(house, path=PATH):
    d = {}
    
    with open(path + f"/house_{house}/labels.dat", 'r') as file:
        for line in file.readlines():
            if not line: continue
        
            d.update({line.split()[0]: line.split()[1]})
            
    return d


def get_max_power(house, channel, path=PATH):
    
    with open(path + f"/house_{house}/labels.dat", 'r') as file:
        line = file.readlines()[channel - 1]
        
    appliance = line.split()[1]
    
    return POWER_DICT[appliance]


def get_rectangle_width(house, channel, path = PATH):

    return 100
    
    with open(path + f"/house_{house}/labels.dat", 'r') as file:
        line = file.readlines()[channel - 1]

    appliance = line.split()[1]

    return WIDTH_DICT[appliance]


def read_dat_file(path):
    """
    Reads the data from a channel's .dat file into two numpy arrays: one
    holding the Unix timestamps, the other holding the corresponding data
    """
    
    with open(path, 'r') as file:
        # Put the times and data into numpy arrays
        times, data = np.array(list(map(lambda line: line.split(), file.readlines()))[:-1]).transpose()
        
    # Convert to the appropriate types
    times = times.astype(np.int32)
    data = data.astype(np.float32)
    
    return times, data


def write_dat_file(path, times, data):
    """
    Writes the times and data entries into the .dat file specified by path
    """
    
    with open(path, "w+") as file:
        for i in range(len(data)):
            file.write(str(times[i]) + ' ' + str(round(data[i], 2)) + '\n')
            
            
def read_npy_file(path):
    """
    Reads the data from a .npy file, returning numpy arrays of the Unix
    timestamps and the corresponding data
    """
    
    # First load the array we stored into the file
    data = np.load(path)
    
    # Split it into the actual data and the timestamps
    raw_times = np.copy(data[-2:])
    data.resize((len(data) - 2,))
    
    begin_time, end_time = np.frombuffer(raw_times.tobytes(), dtype=np.int32)
    times = np.arange(begin_time, end_time + 1)
    
    return times, data


def write_npy_file(path, times, data):
    """
    Writes the times and data arrays into the .npy file specified by path
    """
    
    # First put the begin and end timestamps into their own array
    array_times_int   = np.array([times[0], times[-1]])
    # Then make them pretend to be floats to fit into the data array
    array_times_float = np.frombuffer(array_times_int.tobytes(), dtype=np.float32)
    
    data = np.append(data, array_times_float)
    
    np.save(path, data)


def read_file(path, ftype):
    if ftype == "npy":
        return read_npy_file(path)
    elif ftype == "dat":
        return read_dat_file(path)
    else:
        raise ValueError


def write_file(path, times, data, ftype):
    if ftype == "npy":
        return write_npy_file(path, times, data)
    elif ftype == "dat":
        return write_dat_file(path, times, data)
    else:
        raise ValueError


def delete_preprocessed(path=PATH):
    """
    Deletes all preprocessed data anywhere in the given path
    """
    
    for p, _, _ in os.walk(path):
        if os.path.exists(p) and os.path.isdir(p) and ("_preprocessed" in p):
            shutil.rmtree(p)
