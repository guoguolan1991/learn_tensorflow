# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     dataset
   Description:   获取情感语料数据
   Author:        Miller
   date：         2017/9/12 0012
-------------------------------------------------
"""
__author__ = 'Miller'

import pandas as pd


class dataset(object):
    def __init__(self):
        pass

    @staticmethod
    def load_data():
        neg = pd.read_excel('../data/sentiment/neg.xls', header=None, index=None)
        pos = pd.read_excel('../data/sentiment/pos.xls', header=None, index=None)
        return neg, pos

    @staticmethod
    def load_comment():
        comment = pd.read_excel('../data/sentiment/sum.xls')
        return comment
