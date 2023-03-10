# Author: Jian
# -*- coding: utf-8 -*-
# @Author   : Jian
# @Project  : quantum_classifier
# @Time     : 2023/3/7 13:38

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mnist import x_train, y_train, x_test, y_test  # dataset
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:250]
y_test = y_test[:250]
# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

# number of qubits
num_qubits = 4

# params
epoch = 30
batch_size = 512

# device
dev = qml.device('default.qubit.tf', wires=num_qubits)

# amplitude encoding
def amplitude_encoding(inputs):
    qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits), pad_with=True, normalize=True)

# PQC
def qcircuit(params):
    # 3 layers
    for k in range(3):
        qml.RX(params[8 * k + 0], wires=0)
        qml.RX(params[8 * k + 1], wires=1)
        qml.RX(params[8 * k + 2], wires=2)
        qml.RX(params[8 * k + 3], wires=3)
        qml.RZ(params[8 * k + 4], wires=0)
        qml.RZ(params[8 * k + 5], wires=1)
        qml.RZ(params[8 * k + 6], wires=2)
        qml.RZ(params[8 * k + 7], wires=3)
        qml.broadcast(qml.CNOT, wires=range(num_qubits), pattern='ring')
        qml.broadcast(qml.CNOT, wires=range(num_qubits), pattern='ring')

# quantum part
@qml.qnode(dev, interface='tf')
def qclassifier(params, inputs):
    # qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits), pad_with=True, normalize=True)
    amplitude_encoding(inputs)
    qcircuit(params)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

# quantum payer
qlayer = qml.qnn.KerasLayer(qclassifier, {'params': (24, )}, output_dim=num_qubits)

# classical layers
inputlayer = tf.keras.layers.Dense(32, activation='relu', input_shape=(16, ))
dlayer0 = tf.keras.layers.Dense(16, activation='relu')
dlayer1 = tf.keras.layers.Dense(64, activation='relu')
outputlayer = tf.keras.layers.Dense(2, activation='relu')

# hybrid model
hmodel = tf.keras.Sequential([inputlayer, dlayer0, qlayer, dlayer1, outputlayer], name='hybridNN')

# compile
hmodel.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

# fit model
history0 = hmodel.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_test, y_test))
hmodel.summary()

# classical counterpart
# classical layers
inputlayer = tf.keras.layers.Dense(32, activation='relu', input_shape=(16, ))
dlayer0 = tf.keras.layers.Dense(16, activation='relu')
dlayer1 = tf.keras.layers.Dense(4, activation='relu')
dlayer2 = tf.keras.layers.Dense(64, activation='relu')
outputlayer = tf.keras.layers.Dense(2, activation='relu')

# classical model
cmodel = tf.keras.Sequential([inputlayer, dlayer0, dlayer1, dlayer2, outputlayer], name='classicalNN')

# compile
cmodel.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

# fit model
history1 = cmodel.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_test, y_test))
cmodel.summary()

# plot
train_loss0 = history0.history['loss']
val_loss0 = history0.history['val_loss']
train_acc0 = history0.history['accuracy']
val_acc0 = history0.history['val_accuracy']
train_loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']
train_acc1 = history1.history['accuracy']
val_acc1 = history1.history['val_accuracy']
plt.subplot(2, 2, 1)
plt.plot(train_loss0, label='train_loss_hybridNN')
plt.plot(val_loss0, label='val_loss_hybridNN')
plt.title('loss_hybridNN')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(train_acc0, label='train_accuracy_hybridNN')
plt.plot(val_acc0, label='val_accuracy_hybridNN')
plt.title('accuracy_hybridNN')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(train_loss0, label='train_loss_hybridNN')
plt.plot(train_loss1, label='train_loss_classicalNN')
plt.plot(val_loss0, label='val_loss_hybridNN')
plt.plot(val_loss1, label='val_loss_classicalNN')
plt.title('loss')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(train_acc0, label='train_accuracy_hybridNN')
plt.plot(train_acc1, label='train_accuracy_classicalNN')
plt.plot(val_acc0, label='val_accuracy_hybridNN')
plt.plot(val_acc1, label='val_accuracy_classicalNN')
plt.title('accuracy')
plt.legend()
plt.savefig('./fit_res.png')
plt.show()