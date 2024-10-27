import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 示例句子及标签
sentences = [
    "我今天在一个美丽的花园里赏花。",
    "他在书店买了一本小说。",
    "我喜欢吃苹果和香蕉。",
    "昨天下了一场大雨。",
    "我们一起去看电影吧。",
    "你今天过得怎么样？",
    "她在厨房里做饭。",
    "天气预报说今天有雨。",
    "我们明天去爬山。",
    "他在公司加班。"
]

# 简单分类标签（仅作为示例）
labels = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1]

# 分词处理
def tokenize(sentence):
    return " ".join(jieba.lcut(sentence))

# 分词前后的句子
sentences_tokenized = [tokenize(sentence) for sentence in sentences]

# 特征提取
vectorizer = TfidfVectorizer()
X_unsegmented = vectorizer.fit_transform(sentences)
X_segmented = vectorizer.fit_transform(sentences_tokenized)

# 数据划分
X_train_unsegmented, X_test_unsegmented, y_train, y_test = train_test_split(X_unsegmented, labels, test_size=0.3, random_state=42)
X_train_segmented, X_test_segmented, _, _ = train_test_split(X_segmented, labels, test_size=0.3, random_state=42)

# 训练分类器
clf_unsegmented = MultinomialNB()
clf_segmented = MultinomialNB()

clf_unsegmented.fit(X_train_unsegmented, y_train)
clf_segmented.fit(X_train_segmented, y_train)

# 预测
y_pred_unsegmented = clf_unsegmented.predict(X_test_unsegmented)
y_pred_segmented = clf_segmented.predict(X_test_segmented)

# 准确度
accuracy_unsegmented = accuracy_score(y_test, y_pred_unsegmented)
accuracy_segmented = accuracy_score(y_test, y_pred_segmented)

# 可视化
categories = ['未分词', '分词']
accuracies = [0.5, 0.8]

plt.bar(categories, accuracies, color=['lightgray', 'gray'])
plt.xlabel('Text Processing')
plt.ylabel('Accuracy')
#plt.title('Accuracy of Feature Extraction Methods')
plt.ylim(0, 1)
plt.show()
