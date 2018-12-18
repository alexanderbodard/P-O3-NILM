"""
Input shape:
    Conv1D: (None, Rows, Columns, Channels)
    Dense: (None, length)

Flatten between Conv1D and Dense!
"""

from files import read_npy_file
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Flatten
import numpy as np
from rectangles_preprocessing import prep_data
from matplotlib import pyplot as plt


#Length of the input vector.
LENGTH       = 96
PATH         = "P&O3Data/data/low_freq"
THRESHOLD    = 0.5

def create_model(n_conv=1, dense_size=134, n_dense=1, length=LENGTH, optimizer='rmsprop', dense_activation='relu', conv_activation='relu'):
        model = Sequential()
   
        if n_conv > 0:
            model.add(Conv1D(16, 4, strides=1, activation=conv_activation, padding='valid', input_shape=(length, 1)))
        for i in range(n_conv-1):
            model.add(Conv1D(16, 4, strides=1, activation=conv_activation, padding='valid'))
        if n_conv > 0:
            model.add(Flatten())
            model.add(Dense(dense_size, activation=dense_activation))
        else:
            model.add(Dense(dense_size, activation=dense_activation, input_shape=(length,))) 
        
        for i in range(n_dense):
            model.add(Dense(dense_size, activation=dense_activation))
        
        model.add(Dense(3, activation='linear'))
        
        model.compile(loss='mse',
              optimizer=optimizer)
        
        return model

def get_xy(house_channel_tups, length= LENGTH):
    """
    Give a list with tuples of house channel pairs.
    Get x_train and corresponding y_train (labels).
    """
    for index, house_channel_tup in enumerate(house_channel_tups):
        house, channel = house_channel_tup
        if index == 0:
            x_train, y_train = prep_data(house, channel, length, stride=int(length/2))
        else:
            x, y = prep_data(house, channel, length, stride=int(length/2))
            x_train = np.concatenate((x_train, x), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)    
            
    return x_train, y_train

def get_cross_validation(house_channel_tups, length= LENGTH, train_ratio = 0.8):
    """
    Give a list with tuples of house channel pairs.
    Get x_train, corresponding y_train (labels), x_test and corresponding y_test.
    """
    x, y = get_xy(house_channel_tups, length)
    np.random.seed(200)
    
    random_bools =np.random.random(len(x)) < train_ratio
    inverted_bools = np.logical_not(random_bools)
    
    x_train, y_train = x[random_bools], y[random_bools]
    x_test, y_test = x[inverted_bools], y[inverted_bools]
    return x_train, y_train, x_test, y_test

def train_model(x_train, y_train, eps=20, batch_s=32, model=None, length=LENGTH, output=3, save=True, path=PATH, name='my_model', verbose=0):
    """
    Give x_train, y_train, and a model.
    This function will either save or return the trained model.
    """ 
    if model is None:
        print("Please give a valid model.")
        return
    try:
        model.fit(np.expand_dims(x_train, axis=2), y_train,
        epochs=eps,
        batch_size=batch_s,
        verbose=verbose)
    except:
        print('Except reached.')
        model.fit(x_train, y_train,
        epochs=eps,
        batch_size=batch_s,
        verbose=verbose)
        
            
    # Save if asked.
    if save:
        model.save(path + '/rectangle_models/' + name + '.h5')
    else: return model
               
        
def recreate_power(main_power, model_name, path=PATH, length=LENGTH):
    """
    This function takes a np.array, adds zeros until the length is a multiple of LENGTH.
    Next it disaggregates the power of the desired appliance and returns it for the same time interval.
    """
    # Adapt this code if we actually train multiple appliances.
    
    if len(main_power)%length != 0:
        main_power_extended = np.concatenate((main_power, np.zeros(length - (len(main_power)%length))), axis=0)

    # main_power length is a multiple of LENGTH. Make intervals of length LENGTH.
    p = []
    i = 0
    while i < len(main_power_extended): 
        q = []
        for j in range(i, i+length):
            q.append(main_power_extended[j])              
        p.append(q)
        i = i + length

    main_power_intervals = np.array(p)
    
    model = load_model(PATH + '/rectangle_models/' + model_name + '.h5')
    
    try:                                                            # This will only work if model does not contain convolutional filter. Else, except will fix it.
        y_pred = np.transpose(model.predict(main_power_intervals))
    except:
        main_power_intervals = np.expand_dims(main_power_intervals, axis = 2)
        y_pred = np.transpose(model.predict(main_power_intervals))
    
    # Reconstruct appliance power usage.
    start = y_pred[0]
    stop = y_pred[1]
    average = y_pred[2]
    
    i = 0                                                           # i is used to index main_power_extended.
    j = 0                                                           # j is used to index start, stop and average arrays.
    p = []
    while i + length < len(main_power_extended):                    # For each interval of length LENGTH.
        if start[j] < THRESHOLD:                                    # Set start to 0 if start is smaller than treshold. 
            start[j] = 0
        if 1 - stop[j] < THRESHOLD:                                 # Set stop to 1 if stop is bigger than 1-treshold.
            stop[j] = 1
        
        start_index = int(start[j] * length)                        
        stop_index = int(stop[j] * length)
        average_power = average[j]                                  # Multiply with max_power for actual values. Leave like this to compare with expected values.
        
        k = i
        while k < i + start_index:                                  # Set every value before start time to zero.
            p.append(0)
            k += 1
        
        while k < i + stop_index:                                   # Set every value between start and stop to the given average power value.
            p.append(average_power)
            k += 1
        
        while k < i + length:                                       # Set every value after stop time to 0.
            p.append(0)
            k += 1
        i = i + length                                              # Update i to move to next interval.
        j += 1                                                      
    
    prediction = np.array(p)
    return prediction

def plot_model_test(house, channel, file, model_name, n_plots=1, stride=10000, only_channel=False, length=LENGTH):
    """
    Plot graphs of a given house, channel and file. Give model_name
    """
    _, data = read_npy_file(PATH + f"/house_{house}/channel_{channel}_preprocessed/file_{file}.npy") 

    if len(data) <2 * LENGTH:
        return
                           
    # Appliance data
    data_c = data[:int(len(data)/2)]
    # Main data
    data_m = data[int(len(data)/2):]
    
    if not only_channel:
        pred = recreate_power(data_m, model_name, length=length)
    
    #plt.plot(pred[80000:95000]*800)
    plt.plot(pred[50000:70000]*1000)
    plt.xlabel("Tijd (s)")
    plt.ylabel("Verbruik (W)")
    plt.show()
    """
    for i in range(1, n_plots):
        plt.plot(data_c[10000 + i * stride :10000 + (i+1) * stride])
        if not only_channel:
            plt.plot(pred[10000 + i * stride :10000 + (i+1) * stride])
        plt.show()"""
      
"""
Script to plot some results.
for i in range(10):
    print(i)
    plot_model_test(2,9,i, 'Refr_Conv_2')
    print('Next')
    
Script for evolution of length influence.
for i in range(13,5, -1):
    print(i)
    plot_model_test(2,9,5, f'demo_fridge_{i}', length=int(2**i))
    print('Next')
"""
def get_demo_rect():
    for i in range(13,5, -1):
        print(i)
        plot_model_test(2,9,5, f'demo_fridge_{i}', length=int(2**i))
        print('Next')

def save_demo_rect():
    _, data = read_npy_file(PATH + f"/house_{2}/channel_{9}_preprocessed/file_{5}.npy") 
    data_m = data[int(len(data)/2):]
    for i in range(13,5, -1):
        print(i)
        pred = recreate_power(data_m, f'demo_fridge_{i}', length=int(2**i))
        for j in range(len(pred)):
            if pred[j] < 0.1:
                pred[j] = 0
        for z in range(1,2):
            plt.plot(pred[10000 + z * 10000 :9000 + (z+1) * 10000])
            plt.show()
        np.save(PATH + f'/rectangle_demo/demo_fridge_{int(2**i)}.npy', pred)
        print('Next')


