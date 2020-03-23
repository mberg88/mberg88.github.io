# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:56:50 2020

@author: MattiaL
"""

import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from keras.utils import np_utils
from webapp_help_funs import mkdir_fun

import matplotlib.pyplot as plt

#%% user params
if __name__=="__main__":
    '''
    Builds a web app from tensorflow.
    From
    https://towardsdatascience.com/deploying-a-simple-machine-learning-model...
    ...-into-a-webapp-using-tensorflow-js-3609c297fb04
    '''
    plt.close('all')
    
    mkdir_fun(os.path.join(os.getcwd(), 'results'))
    
    #%% plot data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print ("X_train.shape: {}".format(X_train.shape))
    print ("y_train.shape: {}".format(y_train.shape))
    print ("X_test.shape: {}".format(X_test.shape))
    print ("y_test.shape: {}".format(y_test.shape))
    
    plt.subplot(161)
    plt.imshow(X_train[3], cmap = plt.get_cmap('gray'))
    plt.subplot(162)
    plt.imshow(X_train[5], cmap = plt.get_cmap('gray'))
    plt.subplot(163)
    plt.imshow(X_train[7], cmap = plt.get_cmap('gray'))
    plt.subplot(164)
    plt.imshow(X_train[2], cmap = plt.get_cmap('gray'))
    plt.subplot(165)
    plt.imshow(X_train[0], cmap = plt.get_cmap('gray'))
    plt.subplot(166)
    plt.imshow(X_train[13], cmap = plt.get_cmap('gray'))
    
    plt.show()
    
    #%% normalize inputs from 0–255 to 0–1
    x_train = X_train / 255
    x_test = X_test / 255
    
    #%% one-hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = 10
    
    #%% Conv2D
    x_train_deep_model = x_train.reshape((60000, 28, 28, 1)).astype('float32')
    x_test_deep_model = x_test.reshape((10000, 28, 28, 1)).astype('float32')
    
    deep_model = Sequential()
    deep_model.add(Conv2D(30, 
                          (5, 5), 
                          input_shape = (28, 28, 1), 
                          activation = 'relu'))
    deep_model.add(MaxPooling2D())
    deep_model.add(Conv2D(15, 
                          (3, 3), 
                          activation = 'relu'))
    deep_model.add(MaxPooling2D())
    deep_model.add(Dropout(0.2))
    deep_model.add(Flatten())
    deep_model.add(Dense(128, 
                         activation = 'relu'))
    deep_model.add(Dense(50, 
                         activation = 'relu'))
    deep_model.add(Dense(num_classes, 
                         activation = 'softmax'))
    deep_model.compile(loss = 'categorical_crossentropy', 
                       optimizer = 'adam',
                       metrics = ['accuracy'])
    
    # fit model
    deep_model.fit(x_train_deep_model, 
                   y_train, 
                   validation_data = (x_test_deep_model, y_test), 
                   epochs = 30, 
                   batch_size = 200, 
                   verbose = 2)
    
    scores = deep_model.evaluate(x_test_deep_model, y_test, verbose = 0)
    print("Baseline Error: %.2f%%" % (100-scores[1] * 100))
    
    #%% save model
    deep_model.save(os.path.join('results', 'model.h5'))
