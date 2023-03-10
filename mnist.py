# Author: Jian
# -*- coding: utf-8 -*-
# @Author   : Jian
# @Project  : quantum_classifier
# @Time     : 2023/3/6 21:29

from pennylane import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# dtype
x_train = x_train.astype('float')
x_test = x_test.astype('float')

x_train_raw = []
y_train_raw = []
x_test_raw = []
y_test_raw = []

# preserve data with label 0 or 1
for i in range(len(x_train)):
    if y_train[i] == 0: # label = 0
        x_train_raw.append(x_train[i])
        y_train_raw.append(0)
    elif y_train[i] == 1:   # label = 1
        x_train_raw.append(x_train[i])
        y_train_raw.append(1)
for i in range(len(x_test)):
    if y_test[i] == 0: # label = 0
        x_test_raw.append(x_test[i])
        y_test_raw.append(0)
    elif y_test[i] == 1:   # label = 1
        x_test_raw.append(x_test[i])
        y_test_raw.append(1)
# print(len(x_train_raw), len(y_train_raw), len(x_test_raw), len(y_test_raw)) # train: 12665 vali: 2115

# downsample (28, 28) to (16, ) with maxpooling stride = 7
def img_reshape(img):
    downsampled_img = []
    for i in range(4):
        for j in range(4):
            max = 0
            for k in range(7):
                for l in range(7):
                    if img[k + 7 * i][l + 7 * j] >max:
                        max = img[k + 7 * i][l + 7 * j]
            downsampled_img.append(max)
    return downsampled_img
# downsample res test
# img_test = img_reshape(x_train_raw[0])
# img_show = []
# for i in range(4):
#     t = []
#     for j in range(4):
#         t.append(img_test[i * 4 + j])
#     img_show.append(t)
# plt.imshow(img_show, cmap='gray')
# plt.savefig('./downscaled_0.png')
# plt.show()

# downscale the
x_train_downsampled = []
x_test_downsampled = []
for data in x_train_raw:
    x_train_downsampled.append(img_reshape(data))
for data in x_test_raw:
    x_test_downsampled.append(img_reshape(data))

# asarray
x_train = np.array(x_train_downsampled, requires_grad=False)   # penny: requires_grad=False
x_test = np.array(x_test_downsampled, requires_grad=False)
y_train = np.array(y_train_raw, requires_grad=False)
y_test = np.array(y_test_raw, requires_grad=False)

# normalization
x_train = x_train / 255.
x_test = x_test / 255.
