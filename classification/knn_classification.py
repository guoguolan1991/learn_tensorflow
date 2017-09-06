# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     knn_classification
   Description:   knn 文本分类
   Author:        Miller
   date：         2017/9/6 0006
-------------------------------------------------
"""
__author__ = 'Miller'

import sys
from datasets import datasets
from sklearn.neighbors import KNeighborsClassifier

reload(sys)
sys.setdefaultencoding('utf8')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = datasets.load_sklearn_format()
    for x in range(1, 15):
        knnclf = KNeighborsClassifier(n_neighbors=x)
        knnclf.fit(x_train, y_train)
        precisions = knnclf.predict(x_test)
        num = 0
        precisions = precisions.tolist()
        for i, pred in enumerate(precisions):
            if int(pred) == int(y_test[i]):
                num += 1
        print 'K= '+str(x)+', precision_score:' + str(float(num) / len(precisions))
