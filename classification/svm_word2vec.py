# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     svm_word2vec
   Description:   线性svm做分类
   Author:        Miller
   date：         2017/9/2 0005
-------------------------------------------------
"""
__author__ = 'Miller'

import sys
import gensim
import numpy as np
from sklearn.svm import SVC
from datasets import datasets
from setting import VECTOR_DIR
reload(sys)
sys.setdefaultencoding('utf8')


class svm_word2vec(object):
    def __init__(self):
        self.EMBEDDING_DIM = 400
        self.TEST_SPLIT = 0.2
        self.w2v_model = gensim.models.Word2Vec.load(VECTOR_DIR)

    def data_vector(self, datas):
        result = []
        for data in datas:
            words = data.split(' ')
            vector = np.zeros(self.EMBEDDING_DIM)
            word_num = len(words)
            for word in words:
                if unicode(word) in self.w2v_model:
                    vector += self.w2v_model[unicode(word)]
            if word_num > 0:
                vector = vector/word_num
            result.append(vector)
        return result

if __name__ == '__main__':
    train_datas, train_labels, test_datas, test_labels = datasets.load()
    svm_word2vec = svm_word2vec()
    x_train = svm_word2vec.data_vector(train_datas)
    x_test = svm_word2vec.data_vector(test_datas)

    print 'train doc shape: '+str(len(x_train))+' , '+str(len(x_train[0]))
    print 'test doc shape: '+str(len(x_test))+' , '+str(len(x_test[0]))
    y_train = train_labels
    y_test = test_labels

    print 'SVM...'
    svclf = SVC(kernel='linear')
    svclf.fit(x_train, y_train)
    predictions = svclf.predict(x_test)
    num = 0
    preds = predictions.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print 'precision_score:' + str(float(num) / len(preds))





        




