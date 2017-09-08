# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     linear_regression
   Description:   线性回归
   Author:        Miller
   date：         2017/9/8 0008
-------------------------------------------------
"""
__author__ = 'Miller'

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

data_size = 1000
X = np.linspace(-1, 1, data_size)
Y = 0.5 * X + np.random.normal(0, 0.005, (data_size,))

# plt.scatter(X, Y)
# plt.show()

data_split = int(data_size * 0.8)
X_train, Y_train = X[:data_split], Y[:data_split]
X_test, Y_test = X[data_split:], Y[data_split:]

model = Sequential()
model.add(Dense(units=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')

print('training......')

weights = []
W, b = model.layers[0].get_weights()
weights.append(W[0][0])
for step in range(1000):
    cost = model.train_on_batch(X_train, Y_train)
    W, b = model.layers[0].get_weights()
    weights.append(W[0][0])
    if step % 50 == 0:
       print('loss: %f' % cost)

index = range(len(weights))
plt.scatter(index, weights, color='g')
plt.show()

print('testing......')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('loss: %f' % cost)

W, b = model.layers[0].get_weights()
print(W)
print(b)

# plot predict
# Y_pred = model.predict(X_test)
# plt.scatter(X_test, Y_test)
# plt.plot(X_test, Y_pred, 'r')
# plt.show()

