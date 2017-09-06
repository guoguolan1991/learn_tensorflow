# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     svm_doc2vec
   Description:   svm+doc2vec 做分类
   Author:        Miller
   date：         2017/9/6 0006
-------------------------------------------------
"""
__author__ = 'Miller'

import sys
import gensim
from sklearn.svm import SVC
from datasets import datasets
from setting import doc2vec_model
reload(sys)
sys.setdefaultencoding('utf8')


class svm_doc2vec(object):
    def __init__(self):
        self.model = gensim.models.Doc2Vec.load(doc2vec_model)

    def train_doc2vec_model(self, data_path, model_path):
        sentences = gensim.models.doc2vec.TaggedLineDocument(data_path)
        model = gensim.models.Doc2Vec(sentences, size=200, window=5, min_count=5)
        model.save(model_path)
        print 'num of docs: %d' + len(model.docvecs)

    def load_datasets(self):
        x_train = []
        x_test = []
        for idx, docvec in enumerate(self.model.docvecs):
            if idx < 17600:
                x_train.append(docvec)
            else:
                x_test.append(docvec)
        return x_train, x_test

if __name__ == '__main__':
    train_labels = datasets.load_train_labels()
    test_labels = datasets.load_test_labels()
    svm_doc2vec = svm_doc2vec()
    x_train, x_test = svm_doc2vec.load_datasets()
    y_train = train_labels
    y_test = test_labels

    print 'train doc shape: ' + str(len(x_train)) + ' , ' + str(len(x_train[0]))
    print 'test doc shape: ' + str(len(x_test)) + ' , ' + str(len(x_test[0]))

    print '(3) SVM...'

    svc = SVC(kernel='rbf')
    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)
    num = 0
    predictions = predictions.tolist()
    for i, pred in enumerate(predictions):
        if int(pred) == int(y_test[i]):
            num += 1
    print 'precision_score:' + str(float(num) / len(predictions))

