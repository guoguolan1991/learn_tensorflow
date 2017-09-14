# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     SVMClassifier
   Description:   SVM 情感分类
   Author:        Miller
   date：         2017/9/14 0014
-------------------------------------------------
"""
__author__ = 'Miller'

from sklearn.svm import SVC
import numpy as np


class SVMClassifier:
    def __init__(self, train_data, train_labels, best_words, C):
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        self.best_words = best_words
        self.clf = SVC(C=C)
        self.__train(train_data, train_labels)

    def word2vector(self, all_data):
        vectors = []
        for data in all_data:
            vector = []
            for feature in self.best_words:
                vector.append(data.count(feature))
            vectors.append(vector)
        vectors = np.array(vectors)
        return vectors

    def __train(self, train_data, train_labels):
        '''
        双下划线避免子类覆盖
        '''
        print('SVMClassifier is training......')
        train_vectors = self.word2vector(train_data)

        self.clf.fit(train_vectors, np.array(train_labels))

        print('SVMClassifier train over')

    def classify(self, data):
        vector =  self.word2vector([data])

        prediction = self.clf.predict(vector)

        return prediction[0]