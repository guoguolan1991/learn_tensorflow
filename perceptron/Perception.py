# coding:utf-8


class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化perceptron，设置输入参数的个数，以及激活函数
        :param input_num:
        :param activator:
        '''
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        '''
        打印学习到的权重和偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x * w,
                       zip(input_vec, self.weights)), 0.0) + self.bias
        )

    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入一组向量，以及每个向量对应的label；迭代次数、学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        samples = zip(input_vecs, labels)
        for input_vec, label in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照规则更新权重
        '''
        delta = label - output
        self.weights = map(
            lambda (x, w): w + rate * delta * x,
            zip(input_vec, self.weights)
        )
        # 更新bias
        self.bias += rate * delta
