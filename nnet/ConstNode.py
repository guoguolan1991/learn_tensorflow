# coding:utf-8

# 输出恒唯1的节点
from functools import reduce


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''

        :param layer_index: 节点所属层的编号
        :param node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        '''
        添加一个到下游节点的连接
        '''
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0
        )
        self.delta = self.output * (1 - self.output) * downstream_delta

