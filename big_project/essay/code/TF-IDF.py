import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

# 设置 Matplotlib 中文字体
matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 五个相似的句子
sentences = [
    "我喜欢吃苹果",
    "我喜欢吃香蕉",
    "我不喜欢吃苹果",
    "他喜欢吃苹果",
    "我和他都喜欢吃苹果"
]

# 使用 jieba 对句子进行分词
def jieba_tokenizer(text):
    return jieba.lcut(text)

# 使用 TfidfVectorizer 计算 TF-IDF
vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
tfidf_matrix = vectorizer.fit_transform(sentences)

# 获取词汇表
words = vectorizer.get_feature_names_out()

# 将 TF-IDF 矩阵转换为 DataFrame
df = pd.DataFrame(tfidf_matrix.toarray(), columns=words)
print(df)

# 确保 DataFrame 中所有格子都有数值，设置显示格式
#pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 绘制热图，不显示数值
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=False, cmap="YlGnBu", xticklabels=words, yticklabels=[f'Sentence {i+1}' for i in range(len(sentences))])
#plt.title("TF-IDF 热图")
plt.xlabel("词汇")
plt.ylabel("句子")
plt.show()
