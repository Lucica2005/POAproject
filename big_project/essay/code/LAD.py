import os
import re
import jieba
import opencc
from gensim import corpora, models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib
import numpy as np

# 设置 Matplotlib 中文字体
matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取文件夹中的所有文本文件
def read_files_in_directory(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

# 使用 jieba 对文本进行分词，并去除停用词
def jieba_tokenizer(text):
    # 使用 OpenCC 将繁体转换为简体
    converter = opencc.OpenCC('t2s')
    text = converter.convert(text)
    
    # 使用正则表达式去除所有字母、数字以及非中文字符，只保留中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    
    # 使用 jieba 进行分词
    tokens = jieba.lcut(text)
    
    # 定义停用词列表
    stopwords = {'的', '吗', '了', '呢', '吧', '啊', '嘛', '呀'}
    
    # 移除停用词
    tokens = [token for token in tokens if token not in stopwords]
    
    return tokens

# 读取数据并进行分词处理
def preprocess_texts(texts):
    tokenized_texts = [jieba_tokenizer(text) for text in texts]
    return tokenized_texts

# 应用LDA模型进行主题建模
def apply_lda_model(tokenized_texts, num_topics=5, passes=10):
    # 创建词典
    dictionary = corpora.Dictionary(tokenized_texts)
    
    # 创建词袋（Bag-of-Words）模型
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # 应用LDA模型
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    
    return lda_model, corpus, dictionary

# 显示每个主题的词云
def plot_topic_wordclouds(lda_model, num_topics=5):
    for i in range(num_topics):
        plt.figure()
        word_freq = {word: prob for word, prob in lda_model.show_topic(i, 50)}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        #plt.title(f'主题 {i+1}')
        plt.show()

# 输出每个文本的主题占比
def plot_document_topics(lda_model, corpus, num_topics):
    all_topics = np.zeros((len(corpus), num_topics))
    for i, doc_bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_bow)
        for topic_num, prop in doc_topics:
            all_topics[i, topic_num] = prop
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 7))
    width = 0.2  # 柱状图的宽度
    x = np.arange(len(corpus))  # 文档索引
    
    for i in range(num_topics):
        ax.bar(x + i * width, all_topics[:, i], width, label=f'主题 {i+1}')
    
    ax.set_xlabel('文档索引')
    ax.set_ylabel('主题占比')
    #ax.set_title('所有文本的主题占比')
    ax.legend()
    plt.show()

def main():
    directory_path = 'C:\\Users\\xusir\\Desktop\\college\\courses\\AItheory\\big_project\\essay\\code\\arima_data'
    texts = read_files_in_directory(directory_path)
    tokenized_texts = preprocess_texts(texts)
    lda_model, corpus, dictionary = apply_lda_model(tokenized_texts, num_topics=5, passes=10)
    
    # 打印每个主题的前10个词
    for i, topic in lda_model.print_topics(num_words=10):
        print(f'主题 {i}: {topic}')
    
    # 绘制每个主题的词云
    plot_topic_wordclouds(lda_model, num_topics=5)
    
    # 绘制每个文本的主题占比柱状图
    plot_document_topics(lda_model, corpus, num_topics=5)

if __name__ == '__main__':
    main()
