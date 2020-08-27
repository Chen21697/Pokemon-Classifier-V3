#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:05:15 2020

@author: yuwenchen
"""

import numpy as np
import matplotlib.pyplot as plt
from data import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers import  MaxoutDense
from keras import regularizers


#%%

epochs = 300

if __name__ == '__main__':
    model = Sequential()

    model.add(MaxoutDense(512, nb_feature=3, input_dim=6))
    model.add(Dropout(0.4))
    model.add(Dense(units=1024, activation='sigmoid'))
    model.add(Dense(units=3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, batch_size=25, epochs=epochs, validation_split=0.33)

#%% 
    # for plotting loss
    fig = plt.figure()
    plt.plot(range(0,epochs), history.history['loss'], color='r', label='training loss')
    plt.plot(range(0,epochs), history.history['val_loss'], color='b', label='validation loss')
    plt.xlabel("epoch")

    plt.legend()
    
#%% 
    fig = plt.figure()
    plt.plot(range(0,epochs), history.history['acc'], color='r', label='training acc')
    plt.plot(range(0,epochs), history.history['val_acc'], color='b', label='validation acc')
    plt.xlabel("epoch")

    

    plt.legend()