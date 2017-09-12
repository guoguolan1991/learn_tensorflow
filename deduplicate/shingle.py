# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     shingle
   Description:   用于短文本去重去相似
   Author:        Miller
   date：         2017/9/5 0005
-------------------------------------------------
"""
__author__ = 'Miller'

from deduplicate.ScoreThread import ScoreThread
import time


def string2set(title, win_size=4):
    '''
    文本转为set
    '''
    index = len(title)
    result = set()
    for i in range(index - win_size):
        tmp = title[i:i+win_size]
        result.add(hash(tmp))
    return result


def getscore(text, text_sets):
    max_score = 0.0
    for item in text_sets:
        result = text.intersection(item)
        try:
            tmp = float(len(result)) / max(len(text), len(item))
            max_score = tmp if tmp > max_score else max_score
        except ZeroDivisionError:
            print('text: %d, text_sets: %d' & (len(text), len(item)))
    return max_score


if __name__ == '__main__':
    mt = ScoreThread()
    g_func_list = []
    text = '“用电子围栏拴住共享单车”记者现场实测：有没有都是一个样'
    text2 = '用电子围栏拴住共享单车记者现场实测有没有都是一个样的'
    text3 = '用电子围栏拴住共享单车？实测：有没有都是一个样'
    text_list = []
    text_sets = []
    num_list = [(3000, text), (3000, text2), (4000, text3)]
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for num, text in num_list:
        for i in range(num):
            text_list.append(text)
            text_sets.append(string2set(text))

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    text4 = '电子围栏两套技术标准价差巨大 不同城市各有打算'
    text_set = string2set(text4)
    g_func_list.append({"func": getscore, "args": (text_set, text_sets[:3000])})
    g_func_list.append({"func": getscore, "args": (text_set, text_sets[3000:6000])})
    g_func_list.append({"func": getscore, "args": (text_set, text_sets[6000:])})
    for i in range(10):
        mt.set_thread_func_list(g_func_list)
        print(mt.start())
        # print "max score form all thread: %f" % mt.max_score
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("max score form all thread: %f" % mt.max_score)
