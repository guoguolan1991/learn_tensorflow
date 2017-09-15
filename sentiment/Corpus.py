# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     Corpus
   Description:
   Author:        Miller
   date：         2017/9/14 0014
-------------------------------------------------
"""
__author__ = 'Miller'
import os
import re


class Corpus:
    def __init__(self, filepath):
        root_path = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.normpath(os.path.join(root_path, filepath))

        re_split = re.compile('\s+')

        self.pos_doc_list = []
        self.neg_doc_list = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                splits = re_split.split(line.strip())
                if splits[0] == 'pos':
                    self.pos_doc_list.append(splits[1:])
                elif splits[0] == 'neg':
                    self.neg_doc_list.append(splits[1:])
                else:
                    raise ValueError('Corpus error')

        self.doc_length = len(self.pos_doc_list)
        assert len(self.neg_doc_list) == self.doc_length

        self.train_num = 0
        self.test_num = 0
        runout_content = 'your are using the corpus: %s.\n' % filepath
        runout_content += 'there are total %d positive and %d negative f_corpus.' % (self.doc_length, self.doc_length)
        print(runout_content)

    def get_corpus(self, start=0, end=-1):
        assert self.doc_length >= self.test_num + self.train_num

        if end == -1:
            end = self.doc_length

        data = self.pos_doc_list[start:end] + self.neg_doc_list[start:end]
        data_labels = [1] * (end - start) + [0] * (end - start)
        return data, data_labels

    def get_train_corpus(self, num):
        self.train_num = num
        return self.get_corpus(end=num)

    def get_test_corpus(self, num):
        self.test_num = num
        return self.get_corpus(start=self.train_num, end=self.train_num + num)

    def get_all_corpus(self):
        data = self.pos_doc_list[:] + self.neg_doc_list[:]
        data_labels = [1] * self.doc_length + [0] * self.doc_length
        return data, data_labels


class WaimaiCorpus(Corpus):
    def __init__(self):
        Corpus.__init__(self, '../data/sentiment/corpus/ch_waimai_corpus.txt')

