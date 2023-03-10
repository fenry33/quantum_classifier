# Author: Jian
# -*- coding: utf-8 -*-
# @Author   : Jian
# @Project  : quantum_classifier
# @Time     : 2023/3/6 21:22

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from pennylane.optimize import AdamOptimizer
from mnist import x_train, y_train, x_test, y_test  # dataset
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:250]
y_test = y_test[:250]
# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# params
epoch = 30
batch_size = 125

# number of qubits
num_qubits = 4

# params initializaiton
params_init = np.random.uniform(0, 2 * np.pi, 24, requires_grad=True)

# device
dev = qml.device('default.qubit.tf', wires=num_qubits)

# amplitude encoding
def amplitude_encoding(inputs):
    qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits), pad_with=True, normalize=True)

# PQC   # 24 params
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

# quantum classifier with amplitude encoding
@qml.qnode(dev, interface='tf', diff_method='backprop')  # diff_method='adjoint'
def qclassifier(inputs, params):
    amplitude_encoding(inputs)
    qcircuit(params)
    # measure the last qubit
    # return [qml.expval(qml.PauliZ(wires=3))]  # pennylane training
    return [qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))]   # 2 outputs for one-hot encoding

# PQC visualization
# fig, ax = qml.draw_mpl(qclassifier)(x_train[0], params_init)
# plt.savefig('./topology of qclassifier.png')
# plt.show()

# predictions

# training with pennylane
# def prediciton(params, dataset):
#     pres = []
#     for data in dataset:
#         pres.append(qclassifier(data, params))
#     return pres
#
# # binary cross entropy
# def loss(labels, pres):
#     c = 0.
#     for i, j in zip(labels, pres):
#         p0 = (j + 1) / 2.
#         p1 = 1 - p0
#         c += -1 * np.log(p1 + 1e-15) # 0 * np.log(p0 + 1e-15) + 1 * np.log(p1 + 1e-15)
#         # c += (i - j) ** 2 # mseu
#     return c / (2 * len(labels))
#
# # loss function
# def loss_function(params, dataset, labels):
#     pres = prediciton(params, dataset)
#     return loss(labels, pres)
#
# # accuracy
# def accuracy(labels, pres):
#     a = 0.
#     for i, j in zip(labels, pres):
#         p0 = (j + 1) / 2
#         if i == 0 and p0 > 1 / 2:
#             a += 1
#         elif i == 1 and (1 - p0) > 1 / 2:
#             a += 1
#     return a / len(labels)
#
# # training
# opt = qml.AdamOptimizer(stepsize=0.01)
# epoch = 100
# batch_size = 64
# params = params_init
# loss_histroy = []
# acc_history = []
# for i in range(epoch):
#     batch_index = np.random.randint(0, len(x_train), (batch_size, ))
#     x_batch = x_train[batch_index]
#     y_batch = y_train[batch_index]
#     params, _, _ = opt.step(loss_function, params, x_batch, y_batch)
#     pres = prediciton(params, x_train)
#     cost = loss(y_train, pres)
#     acc = accuracy(y_train, pres)
#     print('iterations:', i+1, '\tloss:', cost, '\taccuracy:', acc)
#     loss_histroy.append(cost)
#     acc_history.append(acc)
#
# # plot
# plt.style.use('seaborn')
# x = range(epoch)
# plt.plot(x, loss_histroy, label='training_loss')
# plt.plot(x, acc_history, label='training_acc')
# plt.legend()
# plt.savefig('./qclassifier_training.png')
# plt.show()

# training with tf
qlayer = qml.qnn.KerasLayer(qclassifier, {'params': (24, )}, output_dim=2)

# model
qmodel = tf.keras.Sequential([qlayer], kernel_initializer=tf.initializers.RandomNormal(0.2))

# compile
qmodel.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

# fit model
history = qmodel.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_test, y_test))
qmodel.summary()

# plot
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['accuracy']
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='train_accuracy')
plt.plot(val_accuracy, label='val_accuracy')
plt.legend()
plt.savefig('./amplitude_encoding_classifier.png')
plt.show()