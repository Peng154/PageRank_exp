import numpy as np
import math

class Tf_idfTransformer(object):
    def wordsToMatrix(self, words_list):
        """
        传入分好词的list，每个元素都是一篇文章
        每个元素中的词用空格隔开
        :param words: 词矩阵
        :return: tf_matix 词频统计矩阵, words 词特征列表
        """
        words = set()

        # 构造词特征列表
        for words_str in words_list:
            seg_list = words_str.split(" ")
            for seg in seg_list:
                words.add(seg)

        words = list(words)

        # 构造词与索引的映射
        words_idx = {}
        i = 0
        for word in words:
            words_idx[word] = i
            i += 1

        # 词频矩阵
        tf_matrix = np.zeros((len(words_list), len(words)), dtype=np.float32)
        # 计算词频
        for i in range(len(words_list)):
            seg_list = words_list[i].split(" ")

            # 单个句子词的总数
            sum = len(seg_list)
            # 单个句子中某个词的出现次数
            words_dict = {}
            # 统计
            for seg in seg_list:
                if seg in words_dict:
                    words_dict[seg] += 1
                else:
                    words_dict[seg] = 1

            # 当前句子仅有的那些词
            keys = words_dict.keys()
            # 计算词频
            for key in keys:
                tf_matrix[i][words_idx[key]] = words_dict[key]/float(sum)

        return tf_matrix, words


    def cal_tf_idf(self, tf_matrix, words, word_list):
        """
        传入词频矩阵，计算tf_idf
        :param tf_matrix: 词频矩阵
        :param words: 词特征向量
        :return: tf_idf_matrix, tf_idf矩阵
        """

        tf_idf_matrix = tf_matrix.copy()
        # 句子总数
        sentences_num = tf_matrix.shape[0]
        print(sentences_num)
        # 词在多少个句子中出现
        word_stc_count = {}

        word_matrix = []
        for word_str in word_list:
            seg_list = word_str.split(" ")
            word_matrix.append(seg_list)

        print(1)
        # 对于每一个词计算出现文档的数量
        for word in words:
            if word not in word_stc_count:
                word_stc_count[word] = 1

            # 如果词语在句子里面，数量加一
            for list in word_matrix:
                if word in list:
                    word_stc_count[word] += 1

        print(2)
        # 构造词与索引的映射
        words_idx = {}
        i = 0
        for word in words:
            words_idx[word] = i
            i += 1

        print(3)
        for i in range(sentences_num):
            for word in words:
                tf_idf_matrix[i][words_idx[word]] *= math.log10(sentences_num/ float(word_stc_count[word]))

        return tf_idf_matrix