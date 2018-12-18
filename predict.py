
import numpy as np
import re
import os
from files import read_npy_file
import matplotlib.pyplot as plt

PATH    = "../../../Data/REDD/data/low_freq"                   # Path to the low_freq folder of the REDD dataset

"""
def predict(model, xs, weights=None, model_predictions=None):
    length = model.input_shape[1]
    
    if weights is None:
        weights = np.ones(length)
    
    l_xs = len(xs)
    
    if model_predictions is None:
        model_predictions = np.array([xs[i: i + LENGTH] for i in range(l_xs - length + 1)])
        model_predictions = model.predict(np.expand_dims(model_predictions, axis=2))
    
    predictions = map(lambda i: np.sum(model_predictions[i - j][j] * weights[j] for j in range(max(0, i - l_xs + length), min(length, i + 1))) / (min(length, i + 1) - max(0, i - l_xs + length)), range(l_xs))

    return np.array(list(predictions))
"""

def predict(model, xs, weights=None, model_predictions=None):
    length = model.input_shape[1]
    
    if weights is None:
        weights = np.ones(length)
        
    l_xs = len(xs)
    
    if model_predictions is None:
        model_predictions = np.array([xs[i: i + length] for i in range(l_xs - length + 1)])
        model_predictions = model.predict(np.expand_dims(model_predictions, axis=2))
    
    weighted_predictions = model_predictions * weights

    weighted_predictions = np.append(np.zeros((length - 1, length)), np.append(weighted_predictions, np.zeros((length - 1, length)), axis=0), axis=0)
    
    flipped_predictions  = np.flip(weighted_predictions, axis=1)

    predictions = np.array(list(map(lambda i: np.trace(flipped_predictions[i - length + 1:i + 1,:]), range(length - 1, l_xs + length - 1))))

    predictions /= np.append(np.arange(1, length + 1), np.append(np.ones(l_xs - 2 * length) * length, np.arange(length, 0, -1)))

    return predictions


def test_mse(model, xs, ys, plot=True, weights=None):
    predictions = predict(model, xs, weights=weights)
    
    # MICROWAVE:
    #for i in range(len(predictions)):
    #    if predictions[i] < max(predictions) * .6:
    #        predictions[i] = 0
    
    if plot:
        plt.plot(predictions)
        plt.show()
        plt.plot(ys)
        plt.show()
#        plt.plot(np.arange(min(20000, len(predictions) - 10000)) / 3600, predictions[10000:30000] * 800, 'C3')
#        plt.show()
#        plt.plot(np.arange(min(20000, len(predictions))) / 3600, predictions[:20000] * 800, 'C2')
#        plt.show()
        print()
        print()
    
    m = np.mean((predictions - ys)**2)

    return m


def test_mse_channel(model, house, channel, path=PATH, plot=True):
    xss = []
    yss = []
    preprocessed_path = path + f"/house_{house}/channel_{channel}_preprocessed/"
    
    for file in os.listdir(preprocessed_path):
        if re.match(r"^file_[0-9]+.npy$", file) is None:
            continue
        
        _, data = read_npy_file(preprocessed_path + file)
        
        data_c, data_m = np.split(data, 2)
        
        if len(data_c) > model.input_shape[1]:
            xss.append(data_m)
            yss.append(data_c)
    
    mean  = 0
    total = sum(map(len, xss))
    for xs, ys in zip(xss, yss):
        new_mean = test_mse(model, xs, ys, plot=plot)
        mean += new_mean * len(xs) / total
        
    return mean
        

def train_prediction_weights(model, xs, ys, initial_weights=None):
    length = model.input_shape[1]
    
    if initial_weights is None:
        initial_weights = np.ones(length)
    
    ...
