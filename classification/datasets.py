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

