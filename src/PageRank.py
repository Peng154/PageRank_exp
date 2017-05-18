import jieba
import pickle, re, time, os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.decomposition.pca import PCA

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

    def pageToSentences(self):
        """
        把每个页面数据看成一个段落，并将这些页面句子拆分开来
        同时会去掉空格以及所有的标点符号
        :return:
        """
        pattern1 = re.compile(r"。|？|！|!|\?")
        pattern2 = re.compile(r"[\s+－：:\.\!\/_,$%^*()【】\[\]\"\']+|[+—！{}<>《》/\\，。？、~@#￥%…&*（）]+")
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
                if len(str)>12:
                    print("{}、processed sentence:{}".format(i, str))
                    sentence = Sentence(str, i)
                    sentences.append(sentence)

        # 经过分析，发现合并句子比较合适。。。。
        sentences = self.merge_sentences(sentences)

        print("句子总数：{}".format(len(sentences)))
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

    def sentencesToVector(self, sentences=None):
        """
        把句子转换成由词的tf—idf组成的向量，因为维数太大而且几乎都是稀疏向量
        所以决定降维到2000维左右
        结束后将处理得到的tf-idf矩阵存储到 ../data/vector.pkl
        :return:
        """

        if sentences is None:
            sentences = self.loadSentences()

        words = []
        for sentence in sentences:
            seg_list = jieba.lcut(sentence.str, cut_all=False)
            # 去停用词
            seg_list = [word for word in seg_list if word not in self.stopWords]
            words.append(' '.join(seg_list))

        print(words[0])
        print(len(words))

        # 开始转换
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(words))

        words = vectorizer.get_feature_names()
        weights = tfidf.toarray()

        print(len(words))
        print(weights.shape)
        for i in weights[0]:
            if i!= 0:
                print(i)

        print('开始PCA降维。。。。')
        start = time.time()
        pca = PCA(n_components= 2000)
        weights = pca.fit_transform(weights)
        print(sum(pca.explained_variance_ratio_))
        end = time.time()
        print('结束，耗时{}s'.format(end-start))

        print(weights.shape)
        # for i in weights[0]:
        #     if i != 0:
        #         print(i)

        file = open('../data/vectors.pkl', 'wb')
        pickle.dump(weights, file)
        file.close()


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

    def analyze_sentences(self):
        '''分析句子长度、分完词之后词的数目'''
        # 加载句子
        if not hasattr(self, 'sentences'):
            if not os.path.exists(self.SENTENCE_FILE_PATH):
                raise ValueError('还未将原始页面转换成句子！！')
            else:
                self.loadSentence()


        # 句子长度
        lengths = []
        # 每个句子词的数量
        word_nums = []

        for sentence in self.sentences:
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

if __name__ == "__main__":
    dataPath = '../data/pages_content.pkl'
    sentences_path = '../data/sentences.pkl'
    stopWords_path = '../data/stopwords.txt'


    pr = PageRank(dataPath=dataPath, sentences_path=sentences_path, stopWords_path=stopWords_path)
    sentences = pr.pageToSentences()
    pr.sentencesToVector(sentences=sentences)
    # pr.cutWords()
    # pr.analyze_sentences()