# coding:utf-8

from Perception import Perceptron

# 定义线性单元的激活函数
f = lambda x: x


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''
        初始化线性单元，设置输入参数的个数
        '''
        Perceptron.__init__(self, input_num, f)


def get_training_dataset():
    input_vecs = [[5], [3], [8], [1.4], [10.1]]

    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    lu = LinearUnit(1)
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu


if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print linear_unit
    print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4])
    print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([15])
    print 'Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5])
    print 'Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3])
