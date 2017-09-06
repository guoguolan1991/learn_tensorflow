# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     bayes_classification
   Description:
   Author:        Miller
   date：         2017/9/6 0006
-------------------------------------------------
"""
__author__ = 'Miller'

import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from datasets import datasets

reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == '__main__':
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

    print '(3) Naive Bayes...'

    clf = MultinomialNB(alpha=0.01)
    clf.fit(x_train, y_train)
    precisions = clf.predict(x_test)
    num = 0
    precisions = precisions.tolist()
    for i, pred in enumerate(precisions):
        if int(pred) == int(y_test[i]):
            num += 1
    print 'precision_score:' + str(float(num) / len(precisions))
