# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     bayes_classification
   Description:   朴素贝叶斯文本分类
   Author:        Miller
   date：         2017/9/6 0006
-------------------------------------------------
"""
__author__ = 'Miller'

import sys
from sklearn.naive_bayes import MultinomialNB
from datasets import datasets

reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = datasets.load_sklearn_format()

    print 'Naive Bayes...'

    clf = MultinomialNB(alpha=0.01)
    clf.fit(x_train, y_train)
    precisions = clf.predict(x_test)
    num = 0
    precisions = precisions.tolist()
    for i, pred in enumerate(precisions):
        if int(pred) == int(y_test[i]):
            num += 1
    print 'precision_score:' + str(float(num) / len(precisions))
