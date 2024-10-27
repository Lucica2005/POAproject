import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import os
import numpy as np
from PIL import Image
import matplotlib

# 设置 Matplotlib 中文字体
matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def generate_ngrams(text, n, stopwords, strict_stop):
    """
    生成文本的 n-gram
    """
    words = jieba.lcut(text)
    words = [word.upper() for word in words]  # 先转换为大写
    words = [word for word in words if word not in stopwords]  # 停用词过滤
    ngrams = zip(*[words[i:] for i in range(n)])
    ngrams = [ngram for ngram in ngrams if not any(ssword in word for ssword in strict_stop for word in ngram)]  # 对生成的n-gram进行严格停用词过滤
    ngrams = [''.join(ngram) for ngram in ngrams]
    ngrams = [ngram for ngram in ngrams if ngram not in stopwords]
    return ngrams

def extract_keywords(texts, topK=20, ngram_range=(1, 2, 3)):
    """
    提取关键词，支持 n-gram
    """
    combined_text = ' '.join(texts)
    keywords = []
    stopwords = set([line.strip().upper() for line in open('C:\\Users\\xusir\\Desktop\\college\\courses\\AItheory\\big_project\essay\code\\3 hotwordAnalysis\\3 hotwordAnalysis\\src\\cn_stopwords.txt', 'r', encoding='utf-8').readlines()])

    # 手动添加停用词
    custom_stopwords = set([
        "这款车", "这辆车", "开起来","这台车子","款车子", "比亚迪宋", "宋PLUSMINI", "宋PLUS", "PLUS新能源",
        "燃油车", "-I", "不会觉得", "比较高", "比较喜欢",
        "比较", "过程中", "后备箱空间", "后排空间", "乘坐空间", "储物空间", "动力方面","中控大屏", "式尾灯", "挺高","满意地方"
    ])
    stopwords.update(custom_stopwords)

    # 严格停用词
    strict_stop = set(["比亚迪", "车", "【", "】", "0","I"])

    # 生成 n-gram 关键词
    for n in ngram_range:
        ngrams = generate_ngrams(combined_text, n, stopwords, strict_stop)
        # 过滤掉无效的字符
        ngrams = [
            ngram for ngram in ngrams
            if ngram.strip() and not any(c in ngram for c in ['\n', '\r', ' '])
        ]
        keywords += ngrams

    # 过滤掉无效的字符
    keywords = [
        kw for kw in keywords
        if kw.strip() and not any(c in kw for c in ['\n', '\r', ' '])
    ]

    keyword_counts = Counter(keywords)
    most_common_keywords = keyword_counts.most_common(topK)
    return most_common_keywords

def create_wordcloud(keywords, image_name, car_image_path):
    """
    生成词云图
    """
    car_mask = np.array(Image.open(car_image_path))  # 使用汽车形状图片作为遮罩

    wordcloud = WordCloud(font_path='C:\\Users\\xusir\\Desktop\\college\\courses\\AItheory\\big_project\essay\code\\3 hotwordAnalysis\\3 hotwordAnalysis\\src\\simsun.ttc', background_color='white', width=2560, height=1600, mask=car_mask)
    wordcloud.generate_from_frequencies(dict(keywords))
    plt.figure(figsize=(12.8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(image_name, dpi=200, bbox_inches='tight')
    plt.show()

def read_comments_from_json(file_path):
    """
    从JSON文件中读取评论
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    comments = [entry['comment'] for entry in data]
    return comments

def plot_keyword_bar_chart(keywords_dict, title):
    """
    生成热点词汇统计柱状图
    """
    for car_model, keywords in keywords_dict.items():
        keywords, counts = zip(*keywords)
        plt.figure(figsize=(12, 8))
        plt.barh(keywords, counts, color='gray')
        plt.xlabel('Count')
        plt.title(f'{title} - {car_model}')
        plt.gca().invert_yaxis()
        plt.show()

def main():
    param = "C:\\Users\\xusir\\Desktop\\college\\courses\\AItheory\\big_project\essay\code\\3 hotwordAnalysis\\3 hotwordAnalysis\\src\\desktop_{model_name}.json"
    car_model_names = ['song', 'qin']
    all_keywords = {}
    
    for car in car_model_names:
        json_file = param.format(model_name=car)
        comments = read_comments_from_json(json_file)
        keywords = extract_keywords(comments, topK=60, ngram_range=range(2, 5))
        all_keywords[car] = keywords
        print(f"{car} 评论关键词：", keywords)
        car_image_path = f'C:\\Users\\xusir\\Desktop\\college\\courses\\AItheory\\big_project\essay\code\\3 hotwordAnalysis\\3 hotwordAnalysis\\src\\{car}.png'  # 假设汽车形状的图片存储在src文件夹中
        create_wordcloud(keywords, f'wordcloud_{car}.png', car_image_path)
    
    # 生成热点词汇统计柱状图
    plot_keyword_bar_chart(all_keywords, '热点词汇统计')

if __name__ == "__main__":
    main()
