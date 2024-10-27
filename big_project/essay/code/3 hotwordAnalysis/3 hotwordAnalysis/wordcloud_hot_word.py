'''
Author: Wh_Xcjm
Date: 2024-06-14 11:54:26
LastEditor: Wh_Xcjm
LastEditTime: 2024-06-14 12:01:30
FilePath: \大项\3 hotwordAnalysis\wordcloud_hot_word.py
Description: 

Copyright (c) 2024 by WhXcjm, All Rights Reserved. 
Github: https://github.com/WhXcjm
'''
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import opencc
import re
import matplotlib
from PIL import Image
import numpy as np
# 设置 Matplotlib 中文字体
matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def jieba_tokenizer(text):
    # 使用 OpenCC 将繁体转换为简体
    converter = opencc.OpenCC('t2s')
    text = converter.convert(text)
    
    # 使用正则表达式去除所有非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 只保留中文字符
    
    # 使用 jieba 进行分词
    tokens = jieba.lcut(text)
    
    # 定义停用词列表
    stopwords = {'的', '吗', '了', '呢', '吧', '啊', '嘛', '呀','是','我','很','有','不','也','高','最','在','没有','都'
                 ,'还有','就','才','比较','不错','就是','没','说','还'}
    
    # 移除停用词
    tokens = [token for token in tokens if token not in stopwords]
    
    return tokens

def plot_tfidf_scores(scores, words, top_n=20):
    # 取得分最高的前 top_n 个词汇和分数
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    sorted_items = sorted_items[:top_n]
    words, scores = zip(*sorted_items)

    # 创建柱状图
    plt.figure(figsize=(10, 8))
    print(words)
    plt.barh(words, scores, color='#F4E1C1')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Words')
    #plt.title('Top TF-IDF Scores in Text')
    plt.gca().invert_yaxis()  # 逆转Y轴，使得分数最高的条在上方
    plt.show()


        

def generate_wordcloud_subplots(forecast_data, num_points=6):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # 创建一个2x3的子图
    axes = axes.flatten()  # 将axes数组展平，方便索引

    # 遍历每个时间点
    for i in range(6):
        # 创建一个词频字典，对每个时间点生成词云
        word_freq = {word: values[i] for word, values in forecast_data.items() if len(values) > i}
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(word_freq)
        
        # 在对应的子图中显示词云
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis('off')  # 关闭坐标轴
        axes[i].set_title(f'Time Point {i+1}')

    plt.tight_layout()  # 调整子图布局
    plt.show()


def main():
    file_path = 'C:\\Users\\xusir\\Desktop\\college\\courses\\AItheory\\big_project\\essay\\code\\train.txt'
    text_content = read_text_file(file_path)
    if text_content:
        sentences = [text_content]

        vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        words = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()

        tfidf_scores = dict(zip(words, scores))
        sorted_tfidf_scores = dict(sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:50])

        # 加载形状图片
        car_mask = np.array(Image.open('C:\\Users\\xusir\\Desktop\\college\\courses\\AItheory\\big_project\\essay\\latex\\pictures\\2.jpg'))  # 更新为实际图片路径

        # 创建词云对象，使用车形状
        wordcloud = WordCloud(font_path='simhei.ttf', width=800, height=800, background_color='white',
                              mask=car_mask, contour_width=1, contour_color='steelblue')
        wordcloud.generate_from_frequencies(sorted_tfidf_scores)

        # 显示词云图
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        plot_tfidf_scores(tfidf_scores, words)
        

if __name__ == '__main__':
    main()
