from urllib import request, parse
from bs4 import BeautifulSoup
import pickle, re, chardet, time


class Crawler(object):
    
    def __init__(self, word, pages_content_path, urls_path, page_count):
        """
        百度搜索结果爬虫初始化函数
        :param word: 需要爬去的搜索关键字，string类型
        :param pages_content_path: 爬下来的搜索结果链接的网页的内容存放路径
        :param urls_path: 爬下来所有搜索结果链接存放文件的路径
        :param page_count: 需要从百度爬取的搜索结果页面数（一页有10条链接）
        """
        self.pages_num = page_count
        self.pages_content_path = pages_content_path
        self.urls_path = urls_path

        assert word is str
        self.data = {'wd': word,
                'pn': '0'}

        self.headers = {
            'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:53.0) Gecko/20100101 Firefox/53.0',
            'Host': 'www.baidu.com',
            'Connection': 'keep-alive',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3'
        }


    def getUrls(self):
        """
        从百度搜索获取所有相关的链接
        """
        # 存储从百度搜索的结果页面链接
        urls = []
        print('开始获取百度搜索结果的url。。。。')
        for i in range(self.pages_num):
            self.data['pn'] = i * 10
            print(self.data)
            url = r'http://www.baidu.com/s?'
            params = parse.urlencode(self.data)
            # 如果需要使用GET方法来获取页面的话，只能这样添加参数
            url = url + str(params)
            # 只要data参数不为空--》采用POST方式获取页面
            req = request.Request(url=url, headers=self.headers)
            response = request.urlopen(req)
            page = response.read()

            soup = BeautifulSoup(page.decode('utf-8'), "html.parser")

            for j in range(i * 10 + 1, (i + 1) * 10 + 1):
                div = soup.find('div', id=str(j))
                print(div.h3.a['href'])
                # 先存起来
                urls.append(div.h3.a['href'])
            # 暂停一下，别被封了。。。。
            time.sleep(3)
        print('获取完成，一共{}页'.format(self.pages_num))
        # 存储到urls文件
        f = open(self.urls_path, 'wb')
        pickle.dump(urls, f)
        f.close()
        return urls

    def getPageContent(self, urls):
        """
        从获取到的链接得到链接网页的内容
        :return: pages，获取到的网页内容，以list的形式存储，一个元素为一个网页内容
        """
        print('开始爬取网页：')

        pages = []

        count = 0
        del self.headers['Host']
        for url in urls:
            count += 1
            print('爬取：{} {}'.format(count,url))
            try:
                req = request.Request(url=url, headers=self.headers)
                response = request.urlopen(req)
                # 得到原始的bytes
                page = response.read()
                # 检测编码方式
                encoding = chardet.detect(page)
                print(encoding['encoding'])
                # 不要这个ISO什么的编码，会乱码的。。。。
                if encoding['encoding'].startswith('ISO'):
                    continue
                try:
                    # 先解码成unicode
                    page = page.decode(encoding['encoding'])
                    # 在编码成为‘utf-8'
                    page = page.encode('utf-8')
                except Exception as e:
                    # 编码出错，跳过。。。。
                    print('第{}个网页编码转换出错！'.format(count))
                    print(e)
                    continue
                soup = BeautifulSoup(page.decode('utf-8'), "html.parser")

                page_content = ''
                # 找到所有的p标签下面的文字，一般这些都是正文
                for p in soup.find_all('p'):
                    # 去掉字符串头尾的\t和空格
                    temp = p.get_text().strip(' ')
                    temp = temp.strip('\t')
                    temp = temp.strip()
                    # 如果单个p标签中的内容太短，很有可能是广告，去掉
                    if len(temp) > 12:
                        page_content += temp

                # 除去一些字符太短太短的网页
                if(len(page_content)>500):
                    print(page_content)
                    pages.append(page_content)

            except Exception as e:
                print(e)
                continue
        # 写入网页内容文件中
        f = open(self.pages_content_path, 'wb')
        pickle.dump(pages, f)
        f.close()
        return pages

    def preProcessData(self, pages):
        """
        预处理数据
        去掉连续空行、制表符等无用信息。
        :return:
        """
        datas = pages

        # 正则表达式的模式
        pattern1 = re.compile(r'\n+')# 连续空行
        pattern2 = re.compile(r'\t+')# 制表符
        pattern3 = re.compile(r'\r\n+')

        for i in range(len(datas)):
            str = datas[i]
            print("{}、原始数据：".format(i))
            print(str)
            print("{}、开始处理。。。。。".format(i))
            str = re.sub(pattern3, "", str)
            str = re.sub(pattern2, "", str)
            str = re.sub(pattern1, "", str)
            print(str)

            datas[i] = str

        # 写入文件中
        f = open(self.pages_content_path, 'wb')
        pickle.dump(datas, f)
        f.close()

        print("总记录数目：{}条".format(len(datas)))

if __name__ == '__main__':
    PAGES_FILE_PATH = '../data/pages_content.pkl'
    URLS_FILE_PATH = '../data/urls.pkl'
    PAGES_NUM = 75
    SEARCH_WORD = '小米笔记本怎么样'

    crawler = Crawler(word=SEARCH_WORD, pages_content_path=PAGES_FILE_PATH,
                      urls_path=URLS_FILE_PATH, page_count=PAGES_NUM)

    urls = crawler.getUrls()
    pages = crawler.getPageContent(urls=urls)
    crawler.preProcessData(pages)

