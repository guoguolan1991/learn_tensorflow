# coding:utf-8
import struct


class Loader(object):
    def __init__(self, path, count):
        '''
        初始化加载器
        :param path: 数据文件路径
        :param count: 文件中的样本个数
        :return:
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

    def to_int(self, byte):
        return struct.unpack('B', byte)[0]


class ImageLoader(Loader):
    # def __init__(self, path, count):
    #     Loader.__init__(self, path, count)

    def get_picture(self, content, index):
        '''
        从文件中获取图像
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    self.to_int(content[start + i * 28 + j])
                )
        return picture

    def get_one_sample(self, picture):
        '''
        将图像转化为样本的输入向量
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        加载全部数据文件，获取全部样本的输入向量
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)
                )
            )
        return data_set


class LabelLoader(Loader):
    def load(self):
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        '''
        将一个值转为10维标签向量
        '''
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_training_data_set():
    image_loader = ImageLoader('../data/train-images-idx3-ubyte.gz', 60000)
    label_loader = LabelLoader('../data/train-labels-idx1-ubyte.gz', 60000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    image_loader = ImageLoader('../data/t10k-images-idx3-ubyte.gz', 10000)
    label_loader = LabelLoader('../data/t10k-labels-idx1-ubyte.gz', 10000)
    return image_loader.load(), label_loader.load()






