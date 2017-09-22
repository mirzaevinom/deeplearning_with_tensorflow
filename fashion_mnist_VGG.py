#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:03:09 2017

@author: mirzaev.1
"""

from __future__ import print_function, division

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, ZeroPadding2D

from keras.optimizers import SGD, Adamax
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import io, gzip, requests
train_image_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
train_label_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
test_image_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
test_label_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"

def readRemoteGZipFile(url, isLabel=True):
    response=requests.get(url, stream=True)
    gzip_content = response.content
    fObj = io.BytesIO(gzip_content)
    content = gzip.GzipFile(fileobj=fObj).read()
    if isLabel:
        offset=8
    else:
        offset=16
    result = np.frombuffer(content, dtype=np.uint8, offset=offset)    
    return(result)

train_labels = readRemoteGZipFile(train_label_url, isLabel=True)
train_images_raw = readRemoteGZipFile(train_image_url, isLabel=False)

test_labels = readRemoteGZipFile(test_label_url, isLabel=True)
test_images_raw = readRemoteGZipFile(test_image_url, isLabel=False)


enc = OneHotEncoder(sparse=False)

X_train = np.array( train_images_raw ).reshape(-1,28,28,1)
y_train = enc.fit_transform( train_labels.reshape(-1,1) )

enc = OneHotEncoder(sparse=False)

X_test = test_images_raw.reshape(-1,28,28,1)
y_test = enc.fit_transform( test_labels.reshape(-1,1) )



def VGG_like_model(n_epoch=10):
    
    n_classes = y_train.shape[1]
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=X_train.shape[1:]) )
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense( n_classes, activation='softmax'))
    
    #adam = Adamax()
    
    model.compile( loss='categorical_crossentropy', 
                   optimizer='adagrad', 
                   metrics=['accuracy'])
    
    model.summary()
    
    model.fit(X_train, y_train, 
              validation_data = (X_test, y_test), 
              epochs=n_epoch, verbose=1, batch_size=64)
    
    


if __name__=='__main__':
    #MLP_with_dropout()
    import time
    
    start= time.time()
    
    VGG_like_model(n_epoch=1)

    end = time.time()

    
    print('Time elapsed: ', round((end-start)/60,2), ' minutes' )
    
    