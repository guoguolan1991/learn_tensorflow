# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     DictSentiment
   Description:   基于词典的情感分析
   Author:        Miller
   date：         2017/9/19 0019
-------------------------------------------------
"""
from sentiment.basedict.LoadDict import LoadDict
import jieba


class DictSentimet:
    def __init__(self, text):
        self.text = text
        self.negdict, self.posdict, self.notdict, self.plusdict = LoadDict.load_dict()
        self.segment = list(jieba.cut(text))
        self.sentiments = self.predict()

    def predict(self, func=None):
        '''
        计算当前的句子的情感倾向，大于0正向，小于0负向。可使用自定义的函数
        func进行计算
        '''
        score = 0
        if not func:
            for i in range(len(self.segment)):
                if self.segment[i] in self.negdict:
                    if i > 0 and self.segment[i - 1] in self.notdict:
                        score = score + 1
                    elif i > 0 and self.segment[i - 1] in self.plusdict:
                        score = score - 2
                    else:
                        score = score - 1
                elif self.segment[i] in self.posdict:
                    if i > 0 and self.segment[i - 1] in self.notdict:
                        score = score - 1
                    elif i > 0 and self.segment[i - 1] in self.plusdict:
                        score = score + 2
                    elif i > 0 and self.segment[i - 1] in self.negdict:
                        score = score - 1
                    elif i < len(self.segment) - 1 and self.segment[i + 1] in self.negdict:
                        score = score - 1
                    else:
                        score = score + 1
                elif self.segment[i] in self.notdict:
                    score = score - 0.5
            return score
        else:
            return func(self.text)

if __name__ == '__main__':
    ds = DictSentimet('这个东西好难吃')
    print(ds.sentiments)
    ds = DictSentimet('这次的服务还可以，整体比较舒服')
    print(ds.sentiments)
    ds = DictSentimet('这家的冰激凌非常棒，下次还来吃')
    print(ds.sentiments)