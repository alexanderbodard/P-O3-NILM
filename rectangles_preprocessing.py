"""
Get rectangles with the specific width for all appliances.

"""

PATH         = "P&O3Data/data/low_freq"
HOUSES       = [1,2,3,4,5,6]
CHANNELS     = {house: list(range(3, [20, 11, 22, 20, 26, 17][house - 1] + 1)) for house in HOUSES}
THRESHOLD    = 0.07
MAX_GAP      = 5

import os, os.path
from files import get_rectangle_width, read_file
import numpy as np
from files import read_npy_file
import math

# The following functions are for preprocessing the training data and to obtain the requiered labels.

def get_rectangles(houses = HOUSES, channels = CHANNELS, path=PATH, threshold=THRESHOLD, max_gap=MAX_GAP):
    """
    This function creates a new subfolder containing a start, stop and average file for each file of the given channel.
    """
    for house in houses:
        for channel in channels[house]:

            DIR = path + f"/house_{house}/channel_{channel}_preprocessed"
            # If also folders in this directory, add:  if os.path.isfile(os.path.join(DIR, name))
            num_of_files = len([name for name in os.listdir(DIR)])

            for file_num in range(num_of_files):
                times, data = read_file(path + f"/house_{house}/channel_{channel}_preprocessed/file_{file_num}.npy", 'npy')

                # We don't need the mains data.
                data = np.array(data[: int(len(data) / 2)])
                # Keep only data bigger than threshold.
                b = (data > threshold)
                new_data = np.array(data)[b]
                new_times = np.array(times)[b]
                
                if len(new_times) == 0:
                    print("No rectangles for " + f"house_{house}/channel_{channel}" + f" file_{file_num}")
                    continue

                start = [new_times[0]]
                stop = []
                average = []

                for i in range(len(new_times) - 1):
                    if new_times[i+1] - new_times[i] > max_gap:
                        mean = np.mean(new_data[np.argwhere(new_times == start[-1])[0][0] : i])
                                              
                        if not math.isnan(mean):
                            average.append(mean)
                            start.append(new_times[i+1])
                            stop.append(new_times[i])
                        else: print('-----------------------------------here')
                        
                # Append last stop and average value.
                stop.append(new_times[-1])
                mean = np.mean(new_data[np.argwhere(new_times == start[-1])[0][0] : stop[-1]])
                if not math.isnan(mean):
                    average.append(mean)
                else: print('-----------------------------------here @ end')
                start_array = np.array(start)
                stop_array = np.array(stop)
                average_array = np.array(average)  
                print('Final:')
                print(len(start), len(stop), len(average))
                
                if not os.path.exists(path + f"/house_{house}/channel_{channel}_rectangles"):
                    os.mkdir(path + f"/house_{house}/channel_{channel}_rectangles")
                np.save(path + f"/house_{house}/channel_{channel}_rectangles/start_{file_num}.npy" , start_array)
                np.save(path + f"/house_{house}/channel_{channel}_rectangles/stop_{file_num}.npy", stop_array)
                np.save(path + f"/house_{house}/channel_{channel}_rectangles/average_{file_num}.npy", average_array)

def load_data(house, channel, path = PATH):
    
    """
    Load start, stop and average and process to labels file. 
    The data and time files are also returned.
    """

    DIR = path + f"/house_{house}/channel_{channel}_preprocessed"
    # If also other files in this directory, add:  if os.path.isfile(os.path.join(DIR, name))
    num_of_files = len([name for name in os.listdir(DIR) ])

    labels = None
    data = None
    time = None

    for file_num in range(num_of_files):
        # Labels.
        if os.path.exists(path + f"/house_{house}/channel_{channel}_rectangles/start_{file_num}.npy"):
            start = np.load(path + f"/house_{house}/channel_{channel}_rectangles/start_{file_num}.npy")
            stop = np.load(path + f"/house_{house}/channel_{channel}_rectangles/stop_{file_num}.npy")
            average = np.load(path + f"/house_{house}/channel_{channel}_rectangles/average_{file_num}.npy")

            if file_num == 0:
                labels = np.array(list(zip(start, stop, average)))
            else:
                labels = np.concatenate((labels, np.array(list(zip(start, stop, average)))))
        # X_train.
        t, raw_data = read_npy_file(path + f"/house_{house}/channel_{channel}_preprocessed/file_{file_num}.npy")
        d = raw_data[int(len(raw_data) / 2):]

        if file_num == 0:
            data, time = d, t
        else:
            data = np.concatenate((data, d))
            time = np.concatenate((time, t))
    return data, time, labels

def prep_data(house, channel, length, stride, path=PATH):
    
    """
    Prep the data, time and labels to corresponding x_train and y_train for given length (and stride).
    """

    data, time, labels = load_data(house, channel, path)

    # Create t and d arrays containing lists of 100 data points. They are used to create the labels.
    # Create p where we store lists containing tuples of time and data pairs. This will be x_train.
    t_array = []
    p = []
    i = 0
    # X_train
    while i + length < len(time):                                                       # Last elements are most likely not included, but Keras needs same input length for every input.
        t_array.append(time[i:i+length])

        q = []
        for j in range(i, i+length):
            q.append(data[j])              
        p.append(q)

        i = i + stride
    
    # y_train
    label_array = []
    j = 0
    for i in range(len(t_array)):
        start_, stop_ = t_array[i][0], t_array[i][-1]
        time_window = stop_-start_
        while start_ > labels[j][1]:                                                    # Make sure we find the first label where the stop value is bigger than the start value of our window.
            j = j+1
            if j >= np.shape(labels)[0]:
                break

        if not j>= np.shape(labels)[0] and stop_ < labels[j][0]:                        # If our time window stops before the start of this first label, there is no active appliance in this window.
            label_array.append(np.array([0, 0, 0]))                                     # Everything should be zero then.

        elif not j >= np.shape(labels)[0]:
            a = max(start_, labels[j][0])
            b = min(stop_, labels[j][1])
            if a < b:
                label_array.append(np.array([(a-start_)/time_window, (b-start_)/time_window, labels[j][2]]))

            else:
                label_array.append(np.array([(start_-start_)/time_window, (b-start_)/time_window, labels[j][2] ]))

        else:
            label_array.append(np.array([0, 0, 0]))
            j = 0
            
    x_train = np.array(p)
    y_train = np.array(label_array)
    return x_train, y_train


