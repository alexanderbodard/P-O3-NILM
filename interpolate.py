"""
Functions to removes time gaps in a data set by either interpolating (if the
gap is smaller than THRESHOLD) or by splitting the data into different files

Functions with arguments 'house_or_times' and 'channel_or_data' can receive
either a house and channel name as arguments and be supplied with an input_ftype
argument, or have times and data arrays as arguments and an input_ftype argument
of None

An output_ftype argument results in the data that would otherwise be written to
a file being returned from the function
"""

import numpy as np
import os

from files import read_file, write_file


THRESHOLD    = 60                                                              # Time gap threshold for moving data into separate files
PATH         = "C:/users/arneb/P&O3/Data/REDD/data/low_freq"                   # Path to the low_freq folder of the REDD dataset
HOUSES       = [1,2,3,4,5,6]
CHANNELS     = [list(range(1, num_channels + 1)) for num_channels in [20, 11, 22, 20, 26, 17]]
INPUT_FTYPE  = ("dat", "npy")[0]                                               # File type to load the data from or store it in; 
OUTPUT_FTYPE = ("dat", "npy")[1]                                               #  either "dat" for the default REDD data format in a .dat
                                                                               #  file or "npy" for a numpy .npy file holding the array,
                                                                               #  allowing for fast saving to and loading from files

def join_results(arrays):
    """
    Joins the returned results from interpolate_channel together to form a times and a data array
    """
    
    times = np.concatenate(tuple([t[0] for t in arrays]))
    data  = np.concatenate(tuple([t[1] for t in arrays]))
    
    return times, data

                                                                               
def remove_negative_diffs(house_or_times, channel_or_data, path=PATH, input_ftype=INPUT_FTYPE, output_ftype=OUTPUT_FTYPE):
    """
    Removes negative time differences within the data
    """
    
    if input_ftype:
        house, channel = house_or_times, channel_or_data
        times, data = read_file(path + f"/house_{house}/channel_{channel}.{input_ftype}", input_ftype)
    else:
        house = channel = "undefined"
        times, data = house_or_times, channel_or_data
    
    sortinds  = times.argsort()
    new_times = times[sortinds]
    new_data  = data [sortinds]

    if output_ftype:
        write_file(path + f"/house_{house}/channel_{channel}.{output_ftype}", new_times, new_data, output_ftype)
    else:
        return new_times, new_data
    
    
def interpolate(data, times, diff, i):
    """
    Interpolates the data and times between index i and i+1 in data and times with a time gap of diff
    """
    
    data_begin, data_end = data[i], data[i + 1]
    step = (data_end - data_begin) / diff

    new_times = np.arange(times[i] + 1, times[i + 1])
    
    if step:
        new_data = np.arange(1, diff, dtype = np.float32) * step + data_begin
    else:
        new_data = np.ones(diff - 1, dtype = np.float32) * data_begin
    
    return new_times, new_data


def interpolate_channel(house_or_times, channel_or_data, threshold=THRESHOLD, path=PATH, input_ftype=INPUT_FTYPE, output_ftype=OUTPUT_FTYPE):
    """
    Goes over all the data in a given channel, interpolates and splits it
    """
    
    if input_ftype:
        house, channel = house_or_times, channel_or_data
        times, data = read_file(path + f"/house_{house}/channel_{channel}.{input_ftype}", input_ftype)
    else:
        house = channel = "undefined"
        times, data = house_or_times, channel_or_data
    
    times, data = remove_negative_diffs(times, data, input_ftype=None, output_ftype=None)

    # Calculate the differences between the consecutive timestamps
    diffs = np.diff(times)
    # Discard those with difference 1, and add the index in the original array
    l1 = np.array([diffs, np.arange(len(diffs))]).transpose()[(diffs - 1).astype(np.bool)]
    
    arrays = []
    current_times = np.empty(times[-1] - times[0], dtype = np.int32)
    current_data  = np.empty(times[-1] - times[0], dtype = np.float32)
    array_index   = 0
    last_i = -1

    for diff, i in l1:  
        if diff == 0:
            current_times[array_index : array_index + i - last_i] = times[last_i : i]
            current_data [array_index : array_index + i - last_i] = data [last_i : i]
        
            array_index += i - last_i
                
            last_i = i
            continue
            
        
        # Add the elements up to the current index to the current arrays
        current_times[array_index : array_index + i - last_i] = times[last_i + 1 : i + 1]
        current_data [array_index : array_index + i - last_i] = data [last_i + 1 : i + 1]
        
        array_index += i - last_i

        # If the time jump is less than the set threshold, interpolate between them
        if diff <= threshold:
            new_times, new_data = interpolate(data, times, diff, i)

            current_times[array_index : array_index + diff - 1] = new_times
            current_data [array_index : array_index + diff - 1] = new_data
            
            array_index += diff - 1
        
        # Else, start new arrays and save the current ones
        else:
            current_times.resize((array_index,))
            current_data .resize((array_index,))
            
            arrays.append((np.copy(current_times), np.copy(current_data)))
            
            current_times = np.empty(times[-1] - times[i], dtype = np.int32)
            current_data  = np.empty(times[-1] - times[i], dtype = np.float32)
            
            array_index = 0
        
        last_i = i
        
    # Add the final elements to the current array
    to_go = len(times) - i - 1
    current_times[array_index : array_index + to_go] = times[i + 1:]
    current_data [array_index : array_index + to_go]  = data [i + 1:]
    
    array_index += to_go
    
    # Deallocate the unnecessarily allocated space
    current_times.resize((array_index,))
    current_data .resize((array_index,))

    # Save it as well
    arrays.append((current_times, current_data))
    
    if output_ftype:
        # Write all of the data to new files
        for file_num, (times_array, data_array) in enumerate(arrays):
            
            # Make the channel_?_preprocessed directory if it doesn't exist
            if not os.path.exists(PATH + f"/house_{house}/channel_{channel}_preprocessed"):
                os.mkdir(PATH + f"/house_{house}/channel_{channel}_preprocessed")
                
            write_file(path + f"/house_{house}/channel_{channel}_preprocessed/file_{output_ftype}.npy", times_array, data_array, output_ftype)
    else:
        return arrays


def interpolate_all(threshold=THRESHOLD, path=PATH, input_ftype=INPUT_FTYPE, 
                    output_ftype=OUTPUT_FTYPE, houses=HOUSES, channels=CHANNELS):
    """
    Maps the interpolate_channel funtion to all the channels
    """
    
    for house in houses:
        for channel in range(1, channels[house - 1] + 1):
            interpolate_channel(house, channel, threshold=threshold, path=path, input_ftype=input_ftype, output_ftype=output_ftype)
    