# Author: Jian
# -*- coding: utf-8 -*-
# @Author   : Jian
# @Project  : quantum_classifier
# @Time     : 2023/3/8 18:55

import tensorflow as tf
import matplotlib.pyplot as plt
from mnist import x_train, y_train, x_test, y_test
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:250]
y_test = y_test[:250]
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# classical layers
inputlayer = tf.keras.layers.Dense(32, activation='relu', input_shape=(16, ))
dlayer0 = tf.keras.layers.Dense(16, activation='relu')
dlayer1 = tf.keras.layers.Dense(4, activation='relu')
dlayer2 = tf.keras.layers.Dense(64, activation='relu')
outputlayer = tf.keras.layers.Dense(2, activation='relu')

# model
model = tf.keras.Sequential([inputlayer, dlayer0, dlayer1, dlayer2, outputlayer], name='classical NN')

# compile
model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

model.summary()