import numpy as np
import pandas as pd
# obs ny! \/
import itertools
import copy
from operator import itemgetter
import time


import librosa
import os
import glob
import re
import seaborn as sn
import matplotlib.pyplot as plt
import scipy
import time
import collections
#import keras
import random
#from keras.utils import np_utils
#from keras.layers import MaxPooling1D, Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, LSTM, ELU, Bidirectional, Attention
#from keras.layers import Dense, Dropout, Activation, Flatten, CuDNNLSTM
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#import tensorflow as tf
import pickle
#from keras.utils.vis_utils import plot_model
#from tensorflow.keras import datasets, layers, models
#from tensorflow.keras.optimizers import Adam
#from sklearn.metrics import confusion_matrix

def instrument_code(filename):
    """
    Function that takes in a filename and returns instrument based on naming convention
    """
    class_names=['bass', 'brass', 'flute', 'guitar', 
             'keyboard', 'mallet', 'organ', 'reed', 
             'string', 'synth_lead', 'vocal']
    
    for name in class_names:
        if name in filename:
            return class_names.index(name)
    else:
        return None

# https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


## Ny training data

#samples_split = 0.8
k_folds = 5 
rng_seed = 100
file_name = "weyo"
with open("CustomDataFull/traindata10000.pkl", "rb") as f:

    pickle_full = pickle.load(f)

full_folds_train = [[]*5 for i in range(5)]
full_folds_test = [[]*5 for i in range(5)]
full_folds_valid = [[]*5 for i in range(5)]

full_tuple = list(pickle_full.items())

print(type(pickle_full))
print(len(pickle_full))
#print((np.array(full_tuple)).shape)

instr_code = lambda x : (instrument_code(x[0]))

# https://stackoverflow.com/questions/752308/split-list-into-smaller-lists-split-in-half
def split_list(alist):
    half= len(alist) // 2
    return alist[:half], alist[half:]


if k_folds != -1:

    X_train_selection = []
    X_valid_selection = []
    X_test_selection = []

    # https://www.codespeedy.com/group-a-list-by-the-values-in-python/
    full_tuple = [[(x, y) for x,y in z]
       for k,z in  itertools.groupby(full_tuple, key=(instr_code))]
    
    #print((full_tuple[0]))
    print((np.array(full_tuple)).shape)

    ls = [type(item) for item in full_tuple]
    print(ls)
    ls2 = [type(item) for item in full_tuple[0]]
    print("ls2: ")
    print(ls2)

    #text_time = str(time.time())
    text_time = str(1)

    random.seed(rng_seed)
    for instrument_class in full_tuple:

        instrument_class = random.sample(instrument_class, len(instrument_class))
        print((np.array(instrument_class)).shape)
        folds = chunks(instrument_class, int(len(instrument_class) / k_folds))
        print("folds!: ")
        #print((np.array(list(folds))).shape)
        print(instrument_class[0][0])
        folds = list(folds)
        for chunk_i in range(0, len(folds)):
            duplicate = copy.copy(folds)
            test_valid = duplicate.pop(chunk_i)

            train = []
            for val in duplicate:
                train += val

            #train = duplicate
            test, valid = split_list(test_valid)

            print(chunk_i)
            print(len(folds))
            full_folds_train[chunk_i] += train
            full_folds_test[chunk_i] += test
            full_folds_valid[chunk_i] += valid


for chunk_i in range(0, len(full_folds_train)):
    with open('folds/' + str(chunk_i) + "train_" + file_name + text_time, 'wb') as f:
        pickle.dump(full_folds_train[chunk_i], f)
    with open('folds/' + str(chunk_i) + "valid_" + file_name + text_time, 'wb') as f:
        pickle.dump(full_folds_valid[chunk_i], f)
    with open('folds/' + str(chunk_i) + "test_" + file_name + text_time, 'wb') as f:
        pickle.dump(full_folds_test[chunk_i], f)

print("weyo!")

if k_folds == -1:
    # TRAIN
    with open("CustomData/traindata.pkl", "rb") as f:
        X_valid_full = pickle.load(f)

    X_valid = []
    y_valid = []

    for (key, value) in X_valid_full.items():
        X_valid.append(value)
        y_valid.append(instrument_code(key))

    X_valid_numpy = np.array(X_valid)
    y_valid_numpy = np.array(y_valid)
    print(X_valid_numpy.shape)

    # VALID
    with open("CustomData/validdata.pkl", "rb") as f:
        X_valid_full = pickle.load(f)

    X_valid = []
    y_valid = []

    for (key, value) in X_valid_full.items():
        X_valid.append(value)
        y_valid.append(instrument_code(key))

    X_valid_numpy = np.array(X_valid)
    y_valid_numpy = np.array(y_valid)
    print(X_valid_numpy.shape)

    # TEST
    with open("CustomData/testdata.pkl", 'rb') as f:
        X_test_full = pickle.load(f)

    X_test = []
    y_test = []

    for(key, value) in X_test_full.items():
        #temporal_value = np.mean(value, axis = 1)
        X_test.append(value)
        y_test.append(instrument_code(key))
    y_test_numpy = np.asarray(y_test)
    X_test_numpy = np.asarray(X_test)
    print(X_test_numpy.shape)

print(y_test_numpy)
print(y_train_numpy)
print(y_valid_numpy)




