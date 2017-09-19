# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     LoadDict
   Description:
   Author:        Miller
   date：         2017/9/19 0019
-------------------------------------------------
"""
__author__ = 'Miller'
import pandas as pd


class LoadDict:
    def __init__(self):
        pass

    @staticmethod
    def load_dict():
        negdict = []
        posdict = []
        notdict = []
        plusdict = []
        file = pd.read_csv('../../data/sentiment/dict/negdict.txt', header=None, encoding='utf-8')
        for i in range(len(file[0])):
            negdict.append(file[0][i])
        file = pd.read_csv('../../data/sentiment/dict/posdict.txt', header=None, encoding='utf-8')
        for i in range(len(file[0])):
            posdict.append(file[0][i])
        file = pd.read_csv('../../data/sentiment/dict/notdict.txt', header=None, encoding='utf-8')
        for i in range(len(file[0])):
            notdict.append(file[0][i])
        file = pd.read_csv('../../data/sentiment/dict/plusdict.txt', header=None, encoding='utf-8')
        for i in range(len(file[0])):
            plusdict.append(file[0][i])
        return negdict, posdict, notdict, plusdict
