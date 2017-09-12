# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     mlp_classification
   Description:   mlp 多层感知机分类器
   Author:        Miller
   date：         2017/9/12 0012
-------------------------------------------------
"""
__author__ = 'Miller'

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, plot_model
import os
from classification.datasets import datasets
from classification.setting import *

# 限制tf的warning的输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    train_datas, train_labels, test_datas, test_labels = datasets.load()
    all_datas = train_datas + test_datas
    all_labels = train_labels + test_labels
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_datas)
    sequences = tokenizer.texts_to_sequences(all_datas)
    word_index = tokenizer.word_index
    print('found %d unique tokens' % len(word_index))
    data = tokenizer.sequences_to_matrix(sequences, mode='tfidf')
    labels = to_categorical(np.asarray(all_labels))
    print('shape of data tensor:', data.shape)
    print('shape of label tensor:', labels.shape)

    print('split data set')
    train_len = int(len(data) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    validation_len = int(len(data) * (1 - TEST_SPLIT))
    x_train = data[:train_len]
    y_train = labels[:train_len]
    x_val = data[train_len:validation_len]
    y_val = labels[train_len:validation_len]
    x_test = data[validation_len:]
    y_test = labels[validation_len:]
    print('train data: ' + str(len(x_train)))
    print('val data: ' + str(len(x_val)))
    print('test data: ' + str(len(x_test)))

    model = Sequential()
    model.add(Dense(512, input_shape=(len(word_index) + 1,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()
    plot_model(model, to_file='mlp_model.png', show_shapes=True)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    print(model.metrics_names)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
    model.save('../data/model/mlp/mlp.h5')

    print('(6) testing model...')
    print(model.evaluate(x_test, y_test))
