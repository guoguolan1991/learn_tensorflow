# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     ScoreThread
   Description:   使用多线程处理
   Author:        Miller
   date：         2017/9/5 0005
-------------------------------------------------
"""
__author__ = 'Miller'

import threading
mutex = threading.Lock()


class ScoreThread(object):
    def __init__(self, func_list=None):
        self.func_list = func_list
        self.threads = []
        self.max_score = 0

    def set_thread_func_list(self, func_list):
        self.func_list = func_list

    def trace_func(self, func, *args, **kwargs):
        ret = func(*args, **kwargs)
        if mutex.acquire():
            self.max_score = ret if ret > self.max_score else self.max_score
            mutex.release()

    def start(self):
        self.max_score = 0
        self.threads = []
        for func_dict in self.func_list:
            if func_dict['args']:
                new_arg_list = []
                new_arg_list.append(func_dict['func'])
                for arg in func_dict['args']:
                    new_arg_list.append(arg)
                new_arg_tuple = tuple(new_arg_list)
                t = threading.Thread(target=self.trace_func, args=new_arg_tuple)
            else:
                t = threading.Thread(target=self.trace_func, args=(func_dict['func'],))
            self.threads.append(t)

        for thread in self.threads:
            thread.start()

        for thread in self.threads:
            thread.join()

        return self.max_score