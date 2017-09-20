#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:40:32 2017

@author: mirzaev.1

Keras version of the Networks built in convolutional_NN.py and fully_connected_NN.py

*Keras is very user friendly.
*Syntax is very similar to scikit-learn
"""

from __future__ import print_function, division

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D

from keras.optimizers import SGD, Adamax

from tensorflow.examples.tutorials.mnist import input_data as mnist_data



def MLP_with_dropout(n_epoch=10):
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
    model = Sequential()
    model.add( Dense(200, input_dim= 28*28 , activation='relu' ) )
    model.add(Dropout(0.25))
    
    model.add( Dense(100, activation='relu' ) )
    model.add(Dropout(0.25))
    
    model.add( Dense(60, activation='relu' ) )
    model.add(Dropout(0.25))
    #
    model.add( Dense(30, activation='relu' ) )
    model.add(Dropout(0.25))
    
    model.add( Dense(10, activation='softmax' ) )
    
    
    
    sgd = SGD(lr=0.1 ,decay=1e-3, momentum=0.9, nesterov=True)
    
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    X_train = mnist.train.images.reshape(-1,28*28)
    y_train =  mnist.train.labels
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=100 )
    
    
    X_test = mnist.test.images.reshape(-1,28*28)
    y_test = mnist.test.labels
    score= model.evaluate(X_test, y_test)
    print( ' - Test loss: ', round(score[0], 4) , ' - Test acc: ', round(score[-1], 4)  )
    



def convolution_NN(n_epoch=3):
    
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
    
    model = Sequential()
    model.add( Conv2D(6, (6, 6), activation='relu', input_shape=(28, 28, 1), padding='same') )
       
    model.add( Conv2D( 12 , ( 5 , 5) , activation = 'relu' , padding = 'same' )  )
    model.add( MaxPooling2D( pool_size=(2, 2) ) )

    model.add( Conv2D( 24 , ( 4 , 4) , activation = 'relu' , padding = 'same' )  )
    model.add( MaxPooling2D(pool_size=(2, 2) ) )

    
    model.add(Flatten())
    
    model.add( Dense(200, activation='relu' ) )
    model.add(Dropout(0.25))
    
    model.add( Dense(10, activation='softmax' ) )
        
    
    #sgd = SGD(lr=0.1 ,decay=1e-3, momentum=0.9, nesterov=True)
    adam = Adamax(lr=0.01, decay=1e-2)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    X_train = mnist.train.images
    y_train =  mnist.train.labels
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=100 )
    
    
    X_test = mnist.test.images
    y_test = mnist.test.labels
    score= model.evaluate(X_test, y_test)
    print( ' - Test loss: ', round(score[0], 4) , ' - Test acc: ', round(score[-1], 4)  )
    
    
    
if __name__=='__main__':
    #MLP_with_dropout()
    convolution_NN()