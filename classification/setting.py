# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     setting
   Description:   配置文件
   Author:        Miller
   date：         2017/9/6 0006
-------------------------------------------------
"""
__author__ = 'Miller'

sougou_train_news = '../data/sougou/train_contents.txt'
sougou_train_labels = '../data/sougou/train_labels.txt'
sougou_test_news = '../data/sougou/test_contents.txt'
sougou_test_labels = '../data/sougou/test_labels.txt'

sougou_all_news = '../data/sougou/all_contents.txt'

# 维基百科语料库训练的词向量model
VECTOR_DIR = '../data/wiki/wiki.cn.text.jian.model'

doc2vec_model = '../data/doc2vec.model'

encoding = 'utf-8'

# 每条新闻最大长度
MAX_SEQUENCE_LENGTH = 100

# 验证集比例
VALIDATION_SPLIT = 0.16

# 测试集比例
TEST_SPLIT = 0.2

