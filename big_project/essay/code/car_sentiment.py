import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载JSON数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

qinplus_data = load_data('C:\\Users\\xusir\\Desktop\\college\\courses\\\\AItheory\\big_project\\essay\\code\\classified_qin_with_sentiment.json')
songplus_data = load_data('C:\\Users\\xusir\\Desktop\\college\\courses\\\\AItheory\\big_project\\essay\\code\\classified_song_with_sentiment.json')

# 转换为DataFrame
df_qinplus = pd.DataFrame(qinplus_data)
df_songplus = pd.DataFrame(songplus_data)

# 提取相关信息
df_qinplus = df_qinplus[['aspect', 'sentiment']]
df_songplus = df_songplus[['aspect', 'sentiment']]

# 按车型和类别统计正负面评论数并计算百分比
def calculate_percentage(df):
    aspect_sentiment_count = df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
    aspect_sentiment_percentage = aspect_sentiment_count.div(aspect_sentiment_count.sum(axis=1), axis=0) * 100
    return aspect_sentiment_percentage

aspect_sentiment_percentage_qinplus = calculate_percentage(df_qinplus)
aspect_sentiment_percentage_songplus = calculate_percentage(df_songplus)

# 生成百分比柱状图
def plot_sentiment_percentage(aspect_sentiment_percentage_qinplus, aspect_sentiment_percentage_songplus, title):
    fig, ax = plt.subplots(figsize=(12, 8))

    # 设置颜色
    colors_qinplus = ['#2E8B57', '#CD5C5C']  # 深绿和深红
    colors_songplus = ['#98FB98', '#FFB6C1']  # 浅绿和浅红

    # 绘制秦PLUS的柱状图
    aspect_sentiment_percentage_qinplus.plot(kind='bar', stacked=True, color=colors_qinplus, ax=ax, position=1, width=0.4)

    # 绘制宋PLUS 新能源的柱状图
    aspect_sentiment_percentage_songplus.plot(kind='bar', stacked=True, color=colors_songplus, ax=ax, position=0, width=0.4)

    #ax.set_title(title)
    ax.set_xlabel('Aspect')
    ax.set_ylabel('Percentage')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['秦PLUS Positive', '秦PLUS Negative', '宋PLUS 新能源 Positive', '宋PLUS 新能源 Negative'], title='Sentiment')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# 合并两个车型的百分比数据，确保每个aspect都有相同的列
combined_aspects = sorted(set(aspect_sentiment_percentage_qinplus.index).union(set(aspect_sentiment_percentage_songplus.index)))
aspect_sentiment_percentage_qinplus = aspect_sentiment_percentage_qinplus.reindex(combined_aspects).fillna(0)
aspect_sentiment_percentage_songplus = aspect_sentiment_percentage_songplus.reindex(combined_aspects).fillna(0)

# 生成合并的百分比柱状图
plot_sentiment_percentage(aspect_sentiment_percentage_qinplus, aspect_sentiment_percentage_songplus, 'Sentiment Analysis by Aspect (Percentage)')

