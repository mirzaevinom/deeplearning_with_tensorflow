#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:57:22 2017

@author: mirzaev.1

This is an exercise that I did to learn deep learning with tensorflow.
The slides can be found at
https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/

3 Convolutional layers + 1 Fully connected layer + Softmax Layer

The visulaziation of network architechture can be found in the above slides.
"""
from __future__ import print_function, division

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)


#This model has 4 hidden layers.

#Input None: we don't know how many inputs will be given, 28x28x1 grayscale images
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, 6] ,stddev=0.1))
B1 = tf.Variable(tf.ones([6])/10)

W2 = tf.Variable(tf.truncated_normal([5,5,6,12], stddev=0.1))
B2 = tf.Variable(tf.ones([12])/10)

W3 = tf.Variable(tf.truncated_normal([4,4,12,24], stddev=0.1))
B3 = tf.Variable(tf.ones([24])/10)

W4 = tf.Variable(tf.truncated_normal([7*7*24, 200], stddev=0.1))
B4 = tf.Variable(tf.ones([200])/10)

W5= tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)



# The model
#First 3 convolutional layers

#This is I guess pooling coeficient, e.g., 1x1 max-pooling (essentially no pooling)
stride = 1 
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)

#2x2 max-pooling
stride = 2 
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)

#2x2 max-pooling
stride = 2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)


#Fully connected layer
YY = tf.reshape(Y3, [-1, 7*7*24])

#Dropout coef for fully connected layer
pkeep = tf.placeholder(tf.float32)
Y4= tf.nn.relu(tf.matmul(tf.nn.dropout(YY, pkeep), W4) + B4)

#This is to avoid NaNs in the output
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# placeholder for correct labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# loss function (MSE), this might yield NaNs
cross_entropy = tf.reduce_sum((Y_-Y)*(Y_-Y))

#loss function (cross-entropy)
#This is to avoid NaNs in the output
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, learning rate = 0.003

lr = tf.placeholder(tf.float32)

#Standard Gradient Descent Optimizer
#train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

#Supposedly better Optimizer which supports variable learning rates.
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# initialize all the outputs
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


#This will be used in epoch update
n_train =  len(mnist.train.labels)
n_batch = 100
epoch_size = int(n_train/n_batch)

for i in range(10*epoch_size):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(n_batch)
    lrmin = 0.0001
    lrmax=  0.003
    l_rate = lrmin+(lrmax-lrmin)*np.exp(-i/2000)
    train_data={X: batch_X, Y_: batch_Y, lr:l_rate, pkeep:0.75}

    # train
    sess.run(train_step, feed_dict=train_data)
    
    if (i+1) % epoch_size == 0:
        print('Epoch: ', int(i/epoch_size)+1 )
        
        #Train accuracy and loss function at this epoch
        a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print('Train-accuracy: ',a, 'Train-loss: ', c )
        
        #Test accuracy and loss function at this epoch
        test_data={X: mnist.test.images, Y_: mnist.test.labels, pkeep:1.0}
        a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        
        print('Test-accuracy: ',a, 'Test-loss: ', c )
        
        print()