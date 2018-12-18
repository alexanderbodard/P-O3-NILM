
from files import read_npy_file
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Reshape
import os
import re
import numpy as np

PATH        = "C:/users/arneb/P&O3/Data/REDD/data/low_freq"                   # Path to the low_freq folder of the REDD dataset
LENGTH      = 128
STRIDE      = 8
ENCODING    = 48
NUM_FILTERS = 8


def prep_data(houses_channels, max_off_on=1, train_ratio=.8, path=PATH, length=LENGTH, stride=STRIDE):
    xs = np.array([])
    ys = np.array([])
    
    np.random.seed(2000)
    
    for house, channel in houses_channels:
        preprocessed_path = path + f"/house_{house}/channel_{channel}_preprocessed/"
        
        xs_on  = []
        xs_off = []
        ys_on  = []
        ys_off = []
        
        for file in os.listdir(preprocessed_path):
            if re.match(r"^file_[0-9]+.npy$", file) is None:
                continue
            
            times, data = read_npy_file(preprocessed_path + file)
            data_c, data_m = np.split(data, 2)
            
            for i in range(0, len(data_c) - length + 1, stride):
                if np.mean(data_c[i : i + length]) > .1:
                    xs_on.append(np.copy(data_m[i : i + length]))
                    ys_on.append(np.copy(data_c[i : i + length]))
                    
                else:
                    xs_off.append(np.copy(data_m[i : i + length]))
                    ys_off.append(np.copy(data_c[i : i + length]))
           
        xs_on  = np.array(xs_on)
        xs_off = np.array(xs_off)
        ys_on  = np.array(ys_on)
        ys_off = np.array(ys_off)
            
        if not xs_off.size:
            xs = np.append(xs, np.copy(xs_on), axis=0)
            ys = np.append(ys, np.copy(ys_on), axis=0)
            
        elif not xs_on.size:
            continue
            
        else:
            random_bools = np.random.random(len(xs_off)) < (len(xs_on) / len(xs_off) * max_off_on)
            
            if xs.size:
                xs = np.append(xs, np.append(xs_on, xs_off[random_bools], axis=0), axis=0)
                ys = np.append(ys, np.append(ys_on, ys_off[random_bools], axis=0), axis=0)
            else:
                xs = np.append(xs_on, xs_off[random_bools], axis=0)
                ys = np.append(ys_on, ys_off[random_bools], axis=0)
            
    random_bools   = np.random.random(len(xs)) < train_ratio
    inverted_bools = np.logical_not(random_bools)
           
    xs_train, ys_train = xs[random_bools],   ys[random_bools]
    xs_test,  ys_test  = xs[inverted_bools], ys[inverted_bools]
    
    return xs_train, ys_train, xs_test, ys_test


def create_model(encoding=ENCODING, num_filters=NUM_FILTERS, length=LENGTH, dense_activation="relu", optimizer="rmsprop", conv_activation1="relu", conv_activation2="linear"):
    model = Sequential()
    
    model.add(Conv1D(filters=NUM_FILTERS, kernel_size=4, strides=1, activation=conv_activation1, padding="valid", input_shape=(length, 1)))
    model.add(Flatten())
    model.add(Dense((length - 3) * NUM_FILTERS, activation=dense_activation))
    model.add(Dense(int(encoding * length), activation=dense_activation))
    model.add(Dense((length + 3) * NUM_FILTERS, activation=dense_activation))
    model.add(Reshape((length + 3, NUM_FILTERS)))
    model.add(Conv1D(filters=1, kernel_size=4, strides=1, activation=conv_activation2, padding="valid"))
    model.add(Flatten())
    
    model.compile(loss="mse", optimizer=optimizer)
    
    return model


def train(model, xs, ys, epochs=30, batch_size=64, verbose=1):
    model.fit(np.expand_dims(xs, axis=2), ys, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    
def train_new_model(houses_channels, encoding=ENCODING, length=LENGTH, stride=STRIDE, epochs=30, batch_size=64, max_off_on=1):
    model = create_model()
    
    xs, ys, _, _ = prep_data(houses_channels, length=length, stride=stride, max_off_on=max_off_on)
    
    train(model, xs, ys, epochs=epochs, batch_size=batch_size)
    
    return model
    