# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     Test
   Description:
   Author:        Miller
   date：         2017/9/14 0014
-------------------------------------------------
"""
__author__ = 'Miller'

import datetime
from sentiment.SVMClassifier import SVMClassifier
from sentiment.Corpus import WaimaiCorpus
from sentiment.FeatureExtraction import FeatureExtraction
from sentiment.Write2File import *


class Test:
    def __init__(self, type_, train_num, test_num, feature_num, max_iter, C, k, corpus):
        self.type_ = type_
        self.train_num = train_num
        self.test_num = test_num
        self.feature_num = feature_num
        self.max_iter = max_iter
        self.C = C
        self.k = k
        self.parameters = [train_num, test_num, feature_num]

        # get corpus
        self.train_data, self.train_labels = corpus.get_train_corpus(train_num)
        self.test_data, self.test_labels = corpus.get_test_corpus(test_num)

        # feature extraction
        fe = FeatureExtraction(self.train_data, self.train_labels)
        self.best_words = fe.best_words(feature_num)
        self.precisions = [[0, 0], [0, 0], [0, 0]]

    def test_svm(self):
        print("SVMClassifier")
        print("---" * 45)
        print("Train num = %s" % self.train_num)
        print("Test num = %s" % self.test_num)
        print("C = %s" % self.C)

        svm = SVMClassifier(self.train_data, self.train_labels, self.best_words, self.C)

        classify_labels = []
        print("SVMClassifier is testing ...")
        for data in self.test_data:
            classify_labels.append(svm.classify(data))
        print("SVMClassifier tests over.")

        filepath = "../output/SVM-%s-train-%d-test-%d-f-%d-C-%d-%s-lin.xls" % \
                   (self.type_,
                    self.train_num, self.test_num,
                    self.feature_num, self.C,
                    datetime.datetime.now().strftime(
                        "%Y-%m-%d-%H-%M-%S"))

        self.write(filepath, classify_labels, 2)

    def write(self, filepath, classify_labels, i=-1):
        results = get_accuracy(self.test_labels, classify_labels, self.parameters)
        if i >= 0:
            self.precisions[i][0] = results[10][1] / 100
            self.precisions[i][1] = results[7][1] / 100

        Write2File.write_contents(filepath, results)


def test_waimai():

    type_ = 'waimai'
    train_num = 3000
    test_num = 1000
    feature_num = 5000
    max_iter = 500
    C = 150
    k = 13
    k = [1, 3, 5, 7, 9, 11, 13]
    corpus = WaimaiCorpus()
    test = Test(type_, train_num, test_num, feature_num, max_iter, C, k, corpus)
    test.test_svm()

if __name__ == '__main__':
    test_waimai()