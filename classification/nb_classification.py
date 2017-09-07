# coding:utf-8

from sklearn.naive_bayes import MultinomialNB
import os
import random
import jieba


def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    for folder in folder_list:
        x = os.path.isdir(folder)
        if folder != '.DS_Store':
            new_folder_path = os.path.join(folder_path, folder)
            files = os.listdir(new_folder_path)
            # for dir in os.listdir(new_folder_path):
            #     if os.path.isdir(dir):
            #         files.append(dir)

            j = 1
            for file in files:
                if j > 100:
                    break
                with open(os.path.join(new_folder_path, file), 'r') as f:
                    content = f.read()

                word_cut = jieba.cut(content, cut_all=False)
                word_list = list(word_cut)

                data_list.append(word_list)
                class_list.append(folder)
                j += 1

    data_class_list = list(zip(data_list, class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    all_word_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_word_dict.keys():
                all_word_dict[word] += 1
            else:
                all_word_dict[word] = 1

    all_words_tuple_list = sorted(all_word_dict.items(), key=lambda func: func[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)
    all_words_list =list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def MakeWordSet(words_file):
    '''
    读取文件内容，并去重
    '''
    words_set = set()
    with open(words_file, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)

    return words_set


def TextFeatures(train_data_list, test_data_list, feature_words):
    '''
    根据特征词把文本向量化
    '''
    def text_feature(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_feature(text, feature_words) for text in train_data_list]
    test_feature_list = [text_feature(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    '''
    文本特征提取
    '''
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:
            break

        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and \
            1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])

    return feature_words


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':
    folder_path = '../data/news/'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,
                                                                                                        test_size=0.2)
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordSet(stopwords_file)

    test_accuracy_list = []
    deleteNs = range(0, 1000, 20)
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)

    print(test_accuracy_list)

