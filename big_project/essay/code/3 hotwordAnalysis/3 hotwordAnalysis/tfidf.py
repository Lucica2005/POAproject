import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import os
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_ngrams(text, n, stopwords, strict_stop):
    """
    生成文本的 n-gram
    """
    words = jieba.lcut(text)
    words = [word.upper() for word in words]
    words = [word for word in words if word not in stopwords]
    ngrams = zip(*[words[i:] for i in range(n)])
    ngrams = [ngram for ngram in ngrams if not any(ssword in word for ssword in strict_stop for word in ngram)]
    ngrams = [''.join(ngram) for ngram in ngrams]
    ngrams = [ngram for ngram in ngrams if ngram not in stopwords]
    return ngrams

def extract_keywords_tfidf(texts, topK=20, ngram_range=(1, 2, 3)):
    stopwords = [line.strip().upper() for line in open('src\\cn_stopwords.txt', 'r', encoding='utf-8').readlines()]

    custom_stopwords = [
        "这款车", "这辆车", "开起来","这台车子","款车子", "比亚迪宋", "宋PLUSMINI", "宋PLUS", "PLUS新能源",
        "燃油车", "-I", "不会觉得", "比较高", "比较喜欢",
        "比较", "过程中", "后备箱空间", "后排空间", "乘坐空间", "储物空间", "动力方面","中控大屏", "式尾灯", "挺高","满意地方"
    ]
    stopwords.extend(custom_stopwords)

    strict_stop = ["比亚迪", "车", "【", "】", "0","I"]

    ngram_texts = []
    for text in texts:
        ngram_text = []
        for n in ngram_range:
            ngram_text += generate_ngrams(text, n, stopwords, strict_stop)
        ngram_texts.append(' '.join(ngram_text))

    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform(ngram_texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel().tolist()
    tfidf_scores = dict(zip(feature_names, tfidf_scores))

    sorted_tfidf = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    filtered_tfidf = [(word, score) for word, score in sorted_tfidf if word not in strict_stop]

    return filtered_tfidf[:topK]

def create_wordcloud(keywords, image_name, car_image_path):
    car_mask = np.array(Image.open(car_image_path))

    wordcloud = WordCloud(font_path='src\\simsun.ttc', background_color='white', width=2560, height=1600, mask=car_mask)
    wordcloud.generate_from_frequencies(dict(keywords))
    plt.figure(figsize=(12.8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(image_name, dpi=200, bbox_inches='tight')
    plt.show()

def read_comments_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    comments = [entry['comment'] for entry in data]
    return comments

def main():
    param = "src\\desktop_{model_name}.json"
    car_model_names = ['宋PLUS新能源', '秦PLUS']
    for car in car_model_names:
        json_file = param.format(model_name=car)
        comments = read_comments_from_json(json_file)
        keywords = extract_keywords_tfidf(comments, topK=60, ngram_range=range(2,5))
        print(f"{car} 评论关键词：", keywords)
        car_image_path = f'src\\{car}.png'
        create_wordcloud(keywords, f'tfidf_wordcloud_{car}.png', car_image_path)

if __name__ == "__main__":
    main()
