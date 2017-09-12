# coding:utf-8
import random
from functools import reduce
from nnet.Network import Network


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        return map(lambda m: 0.9 if number & m else 0.1, self.mask)

    def denorm(self, vec):
        binary = map(lambda i: 1 if i > 0.5 else 0, vec)
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def mean_square_error(vec1, vec2):
    '''
    计算平方误差
    '''
    return 0.5 * reduce(lambda a, b: a + b,
                            map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                                zip(vec1, vec2)
                            )
                        )


def gradient_check(network, sample_feature, sample_label):

    # 获取网络在当前样本下的每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()

        epsilon = 0.0001
        # 增加一个很小的值，计算网络误差
        conn.weight += epsilon
        error1 = mean_square_error(network.predict(sample_feature), sample_label)

        conn.weight -= 2 * epsilon
        error2 = mean_square_error(network.predict(sample_feature), sample_label)

        expected_gradient = (error2 - error1) / (2 * epsilon)

        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient
        ))


def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8):
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def train(network):
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)


def test(network, data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print('\ttestdata(%u)\tpredict(%u)' % (data, normalizer.denorm(predict_data)))


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0

    print('correct_ratio: %.2f%%' % (correct /256 * 100))


def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    net = Network([8, 3, 8])
    train(net)
    net.dump()
    correct_ratio(net)
