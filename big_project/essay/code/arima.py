import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import opencc
import re
import matplotlib

# 设置 Matplotlib 中文字体
matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def read_files_in_directory(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def generate_wordcloud_subplots(forecast_data, num_points=6):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # 创建一个2x3的子图
    axes = axes.flatten()  # 将axes数组展平，方便索引

    # 遍历每个时间点
    for i in range(num_points):
        # 创建一个词频字典，对每个时间点生成词云
        word_freq = {word: values[i] for word, values in forecast_data.items() if len(values) > i}
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(word_freq)
        
        # 在对应的子图中显示词云
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis('off')  # 关闭坐标轴
        axes[i].set_title(f'Time Point {i+1}')

    plt.tight_layout()  # 调整子图布局
    plt.show()

def jieba_tokenizer(text):
    # 使用 OpenCC 将繁体转换为简体
    converter = opencc.OpenCC('t2s')
    text = converter.convert(text)
    
    # 使用正则表达式去除换行符、所有非文字字符，以及数字
    text = re.sub(r'\s+|[^a-zA-Z\u4e00-\u9fa5]', '', text)
    
    # 使用 jieba 进行分词
    return jieba.lcut(text)

def compute_tfidf_for_all_files(texts):
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
    all_tfidf_scores = []

    for text in texts:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        all_tfidf_scores.append(dict(zip(feature_names, scores)))

    return pd.DataFrame(all_tfidf_scores).fillna(0)



def forecast_tfidf(tfidf_df):
    results = {}
    for column in tfidf_df:
        series = tfidf_df[column].dropna()
        model = ARIMA(series, order=(2, 2, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)  # 预测未来10个时间点
        results[column] = forecast
    return results

def plot_all_words_tfidf(tfidf_df):
    plt.figure(figsize=(14, 8))  # 设置图形的大小
    for column in tfidf_df.columns:
        plt.plot(tfidf_df.index, tfidf_df[column], label=column)  # 为每个词汇绘制一条线

    plt.title('TF-IDF Scores Over Time for All Words')  # 设置图标题
    plt.xlabel('Document/Time')  # 设置x轴标题
    plt.ylabel('TF-IDF Score')  # 设置y轴标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()

def plot_forecast_results(forecast_results):
    plt.figure(figsize=(14, 8))  # 设置图形的大小
    for word, forecast in forecast_results.items():
        plt.plot(range(1, 11), forecast, label=word,linewidth=0.2)  # 为每个词汇绘制一条线，假设有10个预测点

    #plt.title('Forecast TF-IDF Scores for Words')  # 设置图标题
    plt.xlabel('Future Time Points')  # 设置x轴标题
    plt.ylabel('Forecasted TF-IDF Score')  # 设置y轴标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()

def main():
    directory_path = 'C:\\Users\\xusir\\Desktop\\college\\courses\\AItheory\\big_project\\essay\\code\\arima_data'
    texts = read_files_in_directory(directory_path)
    tfidf_df = compute_tfidf_for_all_files(texts)
    forecast_results = forecast_tfidf(tfidf_df)

    # 输出预测结果
    print(forecast_results)
    plot_forecast_results(forecast_results)

if __name__ == '__main__':
    main()



