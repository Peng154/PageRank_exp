import jieba, math, pandas as pd
import pickle, re, time, os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.decomposition.pca import PCA
from src.Tf_idfTransformer import Tf_idfTransformer
from sklearn import preprocessing

Sentence = namedtuple('Sentence',['str', 'paragraph'])

class PageRank(object):


    def __init__(self, dataPath, sentences_path, stopWords_path):
        file = open(dataPath, 'rb')
        self.data = pickle.load(file)
        self.sentences_path = sentences_path
        file.close()
        # 获取停用词
        self.stopWords = self.getStopWords(stopWords_path)

    def getStopWords(self, filePath):
        """
        获取停用词表
        :param filePath:
        :return:
        """
        file = open(filePath, 'rb')
        file.seek(0)
        # 需要排除\r\n。。。。
        stopWords = [line.decode('utf-8')[:-2] for line in file]
        file.close()
        return stopWords

    def pageToSentences(self, useCache=True):
        """
        把每个页面数据看成一个段落，并将这些页面句子拆分开来
        同时会去掉空格以及所有的标点符号
        :return:
        """

        # 判断是否使用缓存
        if useCache:
            # 如果文件存在
            if os.path.exists(self.sentences_path):
                file = open(self.sentences_path, 'rb')
                sentences = pickle.load(file)
                file.close()
                return sentences

        # 缓存文件不存在，乖乖算一次吧
        pattern1 = re.compile(r"。|？|！|!|\?")
        pattern2 = re.compile(r"[\s+－：:\.\!\/_,$%^*()【】\[\]\"\']+|[+—！{}<>《》/\\，。？\?、~@#￥%…&*（）]+")
        sentences = []

        for i in range(len(self.data)):
            # 把最后的换行符去掉
            page = self.data[i][:-1]
            # 分成句子
            print("{}、original data：{}".format(i, page))
            strs = re.split(pattern1, page)

            for j in range(len(strs)):
                str = strs[j]
                # 去掉所有的标点符号

                # print("{}、original sentence：{}".format(i, str))
                str = re.sub(pattern2, "", str)
                # 排除过短的句子
                if len(str) > 12:
                    print("{}、processed sentence:{}".format(i, str))
                    sentence = Sentence(str, i)
                    sentences.append(sentence)

        # 经过分析，发现合并句子比较合适。。。。
        sentences = self.merge_sentences(sentences)

        print("句子总数：{}".format(len(sentences)))

        # 如果使用缓存，缓存一下
        if(useCache):
            file = open(sentences_path, 'wb')
            pickle._dump(sentences, file)
            file.close()

        del self.data

        return sentences



    def merge_sentences(self, sentences=None):
        if sentences is None:
            sentences = self.loadSentences()

        print("开始合并句子，合并前句子数量：{}".format(len(sentences)))

        merged_sentences = []
        ms = ''
        last_page_num = 0
        for sentence in sentences:
            if sentence.paragraph != last_page_num:
                # 如果属于不同的段落，重新初始化
                if len(ms) != 0:
                    temp = Sentence(ms, last_page_num)
                    merged_sentences.append(temp)
                ms = sentence.str
                last_page_num = sentence.paragraph
                continue
            else:
                # 如果属于同一段，根据长度来决定是否合并句子
                ms += sentence.str
                if len(ms) >= 100:
                    temp = Sentence(ms, last_page_num)
                    merged_sentences.append(temp)
                    ms = ''

        print('合并完成，合并后句子数量：{}'.format(len(merged_sentences)))

        # 输出页码看一下正不正确
        # for sentence in merged_sentences:
        #     print(sentence.paragraph)

        return merged_sentences

    def sentencesToVector(self, sentences=None, useCache=True, cachePath='../data/vectors.pkl'):
        """
        把句子转换成由词的tf—idf组成的向量，因为维数太大而且几乎都是稀疏向量
        所以决定降维到2000维左右
        结束后将处理得到的tf-idf矩阵存储到 ../data/vector.pkl
        :return: 降维之后的tf_idf权重向量，每个句子的段落编号paragraph_list
        """

        # 判断是否使用缓存
        if useCache:
            # 缓存文件是否存在
            if os.path.exists(cachePath):
                file = open(cachePath, 'rb')
                weights, paragraph_list = pickle.load(file)
                file.close()
                return weights, paragraph_list

        #不存在或者不用缓存，重新算吧

        if sentences is None:
            sentences = self.loadSentences()

        # 句子段落编号
        paragraph_list = []

        words_list = []
        for sentence in sentences:
            seg_list = jieba.lcut(sentence.str, cut_all=False)
            # 去停用词
            seg_list = [word for word in seg_list if word not in self.stopWords]
            words_list.append(" ".join(seg_list))
            paragraph_list.append(sentence.paragraph)

        # 开始转换为tf_idf矩阵
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        print("开始计算tf—idf...")
        start = time.time()
        tf_idf_matrix = transformer.fit_transform(vectorizer.fit_transform(words_list))
        end = time.time()
        print("耗时：{}s".format(end - start))

        words = vectorizer.get_feature_names()
        tf_idf_matrix = tf_idf_matrix.toarray()

        print(len(words))
        # print(tf_idf_matrix.shape)
        # for i in tf_idf_matrix[0]:
        #     if i!= 0:
        #         print(i)
        #
        # ##########################################
        # #test，自己写的tf_idf
        # transformer = Tf_idfTransformer()
        # # 统计词频
        # tf_matrix, words = transformer.wordsToMatrix(words_list)
        # print("得到词频，开始计算tf—idf...")
        # start = time.time()
        # tf_idf_matrix = transformer.cal_tf_idf(tf_matrix, words, words_list)
        # end = time.time()
        # print("耗时：{}s".format(end-start))
        #
        # print(tf_idf_matrix.shape)

        print(words_list[0])

        for i in range(len(tf_idf_matrix[0])):
            if tf_idf_matrix[0][i] != 0:
                print(words[i], tf_idf_matrix[0][i], sep=' ')

        print('开始PCA降维。。。。')
        start = time.time()
        pca = PCA(n_components=2000)
        weights = pca.fit_transform(tf_idf_matrix)
        print(sum(pca.explained_variance_ratio_))
        end = time.time()
        print('结束，耗时{}s'.format(end - start))

        print(weights.shape)
        # for i in weights[0]:
        #     if i != 0:
        #         print(i)
        # 如果使用缓存，缓存一下
        if(useCache):
            file = open(cachePath, 'wb')
            pickle.dump((weights, paragraph_list), file)
            file.close()
        return weights, paragraph_list

    def cutWords(self):
        """
        句子分词，去停用词，这个等会自己做TF-IDF
        :return:
        """

        temp = ''
        for sentence in self.sentences:
            temp += sentence.str

        print("开始分词,字符串长度：{}".format(len(temp)))
        start = time.time()
        # 使用精确模式分词，适合于文本分析
        seg_list = jieba.lcut(temp, cut_all=False)
        end = time.time()
        print("分词结束，用时：{}s".format(end - start))

        print('去重复前长度：{}'.format(len(seg_list)))
        seg_set = set(seg_list)

        print('去停用词前长度：{}'.format(len(seg_set)))
        seg_list = [word for word in seg_set if word not in self.stopWords]

        print('去停用词后长度：{}'.format(len(seg_list)))
        # 总共有大概14404个不同的词语


    def loadSentences(self):
        if os.path.exists(self.sentences_path):
            file = open(self.sentences_path, 'rb')
            sentences = pickle.load(file)
            file.close()
            return sentences
        else:
            raise RuntimeError("无法加载句子文件{}，可能还未将网页处理成句子格式！！！".format(self.sentences_path))

    def analyze_sentences(self, sentences=None):
        '''分析句子长度、分完词之后词的数目'''
        # 加载句子
        if sentences is None:
            sentences = self.loadSentences()

        # 句子长度
        lengths = []
        # 每个句子词的数量
        word_nums = []

        for sentence in sentences:
            str = sentence.str
            # 统计句子长度
            lengths.append(len(sentence.str))
            # 统计分词后词的数量
            seg_list = jieba.lcut(str, cut_all=False)
            seg_set = set(seg_list)
            # 去停用词
            seg_list = [word for word in seg_set if word not in self.stopWords]
            word_nums.append(len(seg_list))

        lengths = np.array(lengths)
        word_nums = np.array(word_nums)

        print("length: max:{}, min{}, mean{}".format(lengths.max(),lengths.min(),lengths.mean()))
        print("word nums: max:{}, min{}, mean{}".format(word_nums.max(),word_nums.min(),word_nums.mean()))

        # 画出句子长度直方图
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9,6))
        ax0.hist(lengths, bins=50, color='b', range=(90, 500))
        # ax0.xlabel('lengths')
        # ax0.ylabel('numbers')

        # 发现短的句子实在是太多了,所以决定先合并句子，最短长度为100（暂定）
        ax1.hist(word_nums, bins=50, color='b', range=(0, 200))
        # ax1.xlabel('word_counts')
        # ax1.ylabel('numbers')

        fig.subplots_adjust(hspace=0.4)
        plt.show()

    def getNearness(self, vec1, vec2, epsilon=1e-1):
        """
        通过余弦夹角运算得到两个向量的相似度
        :param vec1:
        :param vec2:
        :return: 相似度=（2*pi/(角度))，为什么要用角度的倒数，因为角度越大代表相似度越低
        """
        assert len(vec1.shape) == 1 and len(vec1.shape) == 1
        assert vec1.shape[0] == vec2.shape[0]

        dot_mul = vec1.dot(vec2)
        norm1 = np.sqrt(np.sum(np.square(vec1)))
        norm2 = np.sqrt(np.sum(np.square(vec2)))

        result = dot_mul/(norm1*norm2)

        # 神他妈，为什么会出现这样的情况。。。。
        if result>1:
            print(result)
            result = 1
        elif result<-1:
            print(result)
            result = -1

        result = np.arccos(result)

        if result == 0:
            result = epsilon
        return 1./result


    def page_rank(self, tfidf_weights, paragraph_list, topK=10, epsilon=1e-5, max_step=1e2, d=0.85):
        """
        对所有的句子进行page_rank排序，返回PR值为前 num 个句子的索引值
        :param tfidf_weights: 所有句子的tf-idf特征向量
        :param paragraph_list: 每个句子所属的段落
        :param topK: 前几个句子
        :param epsilon: PR值迭代的收敛误差
        :param max_step: PR算法迭代的最大步数
        :param d: PR参数
        :return: 一个存有PR值前num个句子的索引的list
        """
        sen_nums = tfidf_weights.shape[0]

        # page_rank的初始值
        p = np.ones((sen_nums, ), dtype=np.float32)
        p /= sen_nums
        temp_p = np.zeros((sen_nums, ), dtype=np.float32)

        # 构建图
        graph = np.zeros((sen_nums, sen_nums), dtype=np.float32)

        # 计算邻接图的权重
        """
        构造图，顶点是句子，边与边之间的权重wij为句子Vi与Vi之间的tf*idf，
        PR(Vi)为顶点Vi的PR值，In表示入边集合，Out为出边集合，顶点Vi指向
        Vj当且就当句子Vi在同一段Vj，且Vj在出现在Vi的后面
        """
        for i in range(sen_nums):
            for j in range(sen_nums):
                if j < i and paragraph_list[j] == paragraph_list[i]:
                    graph[i][j] = self.getNearness(tfidf_weights[i], tfidf_weights[j])

        print("max:", graph.max())
        print(graph.min())

        graph = preprocessing.normalize(graph)

        print(graph.max())
        print(graph.min())

        step = 0

        # 开始迭代计算句子的page_rank的值
        while True:
            start = time.time()
            step += 1
            print("步数：{}".format(step))
            for i in range(sen_nums):
                temp = 0
                for j in range(sen_nums):
                    if graph[j][i] != 0:
                        temp+=(graph[j][i] * p[j])/np.sum(graph[j])

                temp_p[i] = (1-d) + d*temp

            error = np.max(np.abs(p-temp_p))

            end = time.time()
            print("误差：{}， 用时：{}s".format(error, end - start))

            if step < max_step:
                if error > epsilon:
                    p = temp_p.copy()
                else:
                    break
            else:
                break

        # 如果不收敛，返回空
        if step == max_step:
            print("无法收敛啊，是不是有毒!!!")
            return None
        else:
            # 排序，返回前面的索引
            df = pd.DataFrame({'value':p,
                               'idx': range(len(p))})
            df = df.sort_values(by='value')

            file = open('../data/result.pkl', 'wb')
            pickle.dump(df, file)
            file.close()

            result_idx = df['idx'].head(topK).get_values()
            return result_idx



if __name__ == "__main__":
    dataPath = '../data/pages_content.pkl'
    sentences_path = '../data/sentences.pkl'
    stopWords_path = '../data/stopwords.txt'


    pr = PageRank(dataPath=dataPath, sentences_path=sentences_path, stopWords_path=stopWords_path)
    sentences = pr.pageToSentences(useCache=True)
    weights, paragraph_list = pr.sentencesToVector(sentences=sentences, useCache=True)
    idxs = pr.page_rank(weights, paragraph_list, topK=20)
    # file = open('../data/result.pkl', 'rb')
    # df = pickle.load(file)
    # result_idx = df['idx'].head(10).get_values()
    count = 1
    for idx in idxs:
        print(count, "、",sentences[idx].str)
        count+=1

    # pr.cutWords()
    # pr.analyze_sentences()
