# coding:utf-8


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
        conn.weight += epsilon







































































