# PageRank_exp
2017华南理工大学（SCUT）本科三年级数据挖掘课程实验一

## 实验目的：
掌握链接分析的基本思想，及著名的PageRank算法。 
## 实验题目：文章句子的PageRank分值计算
- 针对某种产品的评论主题，使用爬虫程序或人工下载一些网页；
- 构造图，顶点是句子，边与边之间的权重wij为句子Vi与Vj之间的tf*idf，PR(Vi)为顶点Vi的PR值，In表示入边集合，Out为出边集合，顶点Vi指向Vj当且就当句子Vi在同一段Vj，且Vj在出现在Vi的后面；或者采用完全无向图。
- 计算这些句子的PageRank值。并在实验报告中按PageRank的值从高到底打印前10个句子。
- PageRank算法见上课的PPT。
