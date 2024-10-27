import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

def jieba_tokenizer(text):
    # 使用 OpenCC 将繁体转换为简体
    converter = opencc.OpenCC('t2s')
    text = converter.convert(text)
    # 使用正则表达式去除换行符、所有非文字字符，以及数字
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用 jieba 进行分词
    return jieba.lcut(text)

def compute_tfidf_for_all_files(texts):
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
    # 先使用所有文本来拟合vectorizer，建立全局词汇表
    vectorizer.fit(texts)
    all_tfidf_scores = []
    # 然后对每个文本转换成TF-IDF向量，这样每个文本的向量长度都相同
    for text in texts:
        tfidf_matrix = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        all_tfidf_scores.append(dict(zip(feature_names, scores)))
    # 转换成DataFrame，行为词汇，列为文本编号
    return pd.DataFrame(all_tfidf_scores).fillna(0).T

def forecast_tfidf(tfidf_df):
    results = {}
    for column in tfidf_df:
        series = tfidf_df[column].dropna()
        model = ARIMA(series, order=(2, 2, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)  # 预测未来10个时间点
        results[column] = forecast
    return results

def plot_forecast_results(forecast_results):
    plt.figure(figsize=(14, 8))
    if not forecast_results:
        print("No data to plot.")
        return
    for word, forecast in forecast_results.items():
        if forecast is not None and not forecast.empty:
            plt.plot(range(1, 11), forecast, label=word, linewidth=1)
    plt.title('Forecast TF-IDF Scores for Words')
    plt.xlabel('Future Time Points')
    plt.ylabel('Forecasted TF-IDF Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_wordcloud_subplots(forecast_data, num_points=3):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes = axes.flatten()
    for i in range(num_points):
        word_freq = {word: values[i] for word, values in forecast_data.items() if len(values) > i and values[i] > 0}
        if word_freq:
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(word_freq)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].axis('off')
            axes[i].set_title(f'Time Point {i+1}')
        else:
            axes[i].text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    directory_path = 'C:\\Users\\xusir\\Desktop\\college\\courses\\AItheory\\big_project\\essay\\code\\arima_data'
    texts = read_files_in_directory(directory_path)
    tfidf_df = compute_tfidf_for_all_files(texts)
    print(tfidf_df)
    forecast_results = forecast_tfidf(tfidf_df)
    print(forecast_results)
    plot_forecast_results(forecast_results)
    generate_wordcloud_subplots(forecast_results)

if __name__ == '__main__':
    main()
