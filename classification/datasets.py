# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     datasets
   Description:   数据加载类
   Author:        Miller
   date：         2017/9/6 0006
-------------------------------------------------
"""
__author__ = 'Miller'
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from setting import *


class datasets(object):
    def __init__(self):
        pass

    @staticmethod
    def load():
        print('load datasets')
        train_datas = open(sougou_train_news).read().split('\n')
        train_labels = open(sougou_train_labels).read().split('\n')
        test_datas = open(sougou_test_news).read().split('\n')
        test_labels = open(sougou_test_labels).read().split('\n')
        return train_datas, train_labels, test_datas, test_labels

    @staticmethod
    def load_train_datas():
        train_datas = open(sougou_train_news).read().split('\n')
        return train_datas

    @staticmethod
    def load_train_labels():
        train_labels = open(sougou_train_labels).read().split('\n')
        return train_labels

    @staticmethod
    def load_test_datas():
        test_datas = open(sougou_test_news).read().split('\n')
        return test_datas

    @staticmethod
    def load_test_labels():
        test_labels = open(sougou_test_labels).read().split('\n')
        return test_labels

    @staticmethod
    def load_all_datas():
        all_datas = open(sougou_all_news).read().split('\n')
        return all_datas

    @staticmethod
    def load_sklearn_format():
        train_datas, train_labels, test_datas, test_labels = datasets.load()
        all_data = train_datas + test_datas
        count_v0 = CountVectorizer()
        counts_all = count_v0.fit_transform(all_data)

        count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)
        counts_train = count_v1.fit_transform(train_datas)
        print "the shape of train is " + repr(counts_train.shape)

        count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
        counts_test = count_v2.fit_transform(test_datas)
        print "the shape of test is " + repr(counts_test.shape)

        tfidftransformer = TfidfTransformer()
        train_data = tfidftransformer.fit(counts_train).transform(counts_train)
        test_data = tfidftransformer.fit(counts_test).transform(counts_test)

        x_train = train_data
        y_train = train_labels
        x_test = test_data
        y_test = test_labels
        return x_train, y_train, x_test, y_test
