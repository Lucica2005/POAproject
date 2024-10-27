import re
import jieba
import opencc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
unique_labels = ["吃饭", "睡觉", "写作业", "锻炼", "编程", "出游", "谈恋爱", "写论文"]
stopwords = {'的', '吗', '了', '呢', '吧', '啊', '嘛', '呀'}
texts = [
    "我想找一家好吃的餐厅吃晚饭", "午餐吃什么比较好", "找个地方吃午饭", "晚餐吃点什么好", "午饭吃什么",
    "今晚我打算早点睡觉", "昨晚睡得很香", "昨晚失眠了", "今天要早点睡觉", "昨晚做了一个好梦",
    "我需要完成数学作业", "今天的英语作业很多", "今晚要赶完作业", "周末有很多作业要写", "今天有很多数学作业",
    "今天下午我要去跑步锻炼", "我每周三去健身房锻炼", "早上去公园锻炼身体", "下午去健身房锻炼", "每天早上跑步锻炼",
    "我正在学习Python编程", "正在调试一个编程项目", "编写一个新的软件功能", "学习新的编程技术", "编写一个新的手机应用",
    "周末计划去郊区出游", "下个月计划去旅游", "准备一次短途出游", "计划去海边旅游", "计划去山里露营",
    "和女朋友去看电影", "和男朋友一起吃晚餐", "计划和女朋友去购物", "和女朋友去看展览", "和男朋友一起做饭",
    "我需要写一篇关于机器学习的论文", "写一篇关于自然语言处理的论文", "准备写一篇学术论文", "完成一篇关于区块链的论文", "准备写一篇关于人工智能的论文"
]

labels = [
    "吃饭", "吃饭", "吃饭", "吃饭", "吃饭",
    "睡觉", "睡觉", "睡觉", "睡觉", "睡觉",
    "写作业", "写作业", "写作业", "写作业", "写作业",
    "锻炼", "锻炼", "锻炼", "锻炼", "锻炼",
    "编程", "编程", "编程", "编程", "编程",
    "出游", "出游", "出游", "出游", "出游",
    "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱",
    "写论文", "写论文", "写论文", "写论文", "写论文"
]

texts += [
    "我想吃川菜", "晚饭吃什么好呢", "找家好的餐厅吃饭", "午餐想吃点好的", "晚餐想吃日料",
    "今天想早睡", "昨晚做了很多梦", "需要补觉", "今天早点上床休息", "昨晚没睡好",
    "数学作业有点难", "英语作业还没做", "作业写不完了", "物理作业很难", "化学作业很多",
    "今天去游泳", "早上跑步很舒服", "去健身房锻炼", "下午打篮球", "傍晚去散步",
    "正在写一个Python脚本", "编写一个新的游戏", "调试代码有点难", "学习新的编程语言", "写一个新的网站",
    "计划去海边玩", "准备去山里露营", "周末去城市游玩", "下个月去旅游", "计划去欧洲旅行",
    "和男朋友约会", "和女朋友逛街", "一起吃晚餐", "和女朋友看电影", "一起去旅游",
    "写一篇关于深度学习的论文", "准备写一篇大数据论文", "写一篇关于人工智能的论文", "写一篇关于区块链的论文", "准备写一篇机器学习的论文"
]

labels += [
    "吃饭", "吃饭", "吃饭", "吃饭", "吃饭",
    "睡觉", "睡觉", "睡觉", "睡觉", "睡觉",
    "写作业", "写作业", "写作业", "写作业", "写作业",
    "锻炼", "锻炼", "锻炼", "锻炼", "锻炼",
    "编程", "编程", "编程", "编程", "编程",
    "出游", "出游", "出游", "出游", "出游",
    "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱",
    "写论文", "写论文", "写论文", "写论文", "写论文"
]

texts += [
    "我想吃火锅", "晚餐吃点什么", "中午吃饭去哪家餐厅", "晚饭吃啥好呢", "午餐吃什么菜好",
    "今天早点睡", "昨晚失眠了一整晚", "需要午睡一会儿", "今天早睡", "昨晚梦见了老朋友",
    "历史作业好难", "地理作业还没完成", "作业好多写不完", "语文作业太多了", "政治作业也很多",
    "今天去健身房", "早上跑步真不错", "去做瑜伽", "下午去打羽毛球", "傍晚去爬山",
    "编写一个新的程序", "编程作业太多了", "写一个自动化脚本", "学习C语言", "开发一个小程序",
    "准备去沙滩度假", "打算去乡下玩", "计划去野营", "下周去爬山", "下个月去看海",
    "和女朋友去吃大餐", "和男朋友去散步", "一起看电影", "和女朋友去游乐园", "和男朋友去逛街",
    "写一篇关于数据科学的论文", "写一篇云计算的论文", "准备写一篇计算机视觉的论文", "写一篇关于神经网络的论文", "写一篇机器翻译的论文"
]

labels += [
    "吃饭", "吃饭", "吃饭", "吃饭", "吃饭",
    "睡觉", "睡觉", "睡觉", "睡觉", "睡觉",
    "写作业", "写作业", "写作业", "写作业", "写作业",
    "锻炼", "锻炼", "锻炼", "锻炼", "锻炼",
    "编程", "编程", "编程", "编程", "编程",
    "出游", "出游", "出游", "出游", "出游",
    "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱",
    "写论文", "写论文", "写论文", "写论文", "写论文"
]
texts += [
    "午饭吃什么", "晚餐想吃披萨", "找个地方吃午餐", "晚上吃自助餐怎么样", "午餐吃意大利面",
    "今晚早点休息", "昨晚睡得很好", "需要多睡一会儿", "今天晚上要早睡", "昨晚做了个美梦",
    "完成化学作业", "物理作业很多", "今晚要赶完所有作业", "周末还有作业要写", "今天的数学作业很难",
    "下午去游泳", "早上晨跑很舒服", "去健身房举重", "下午去打网球", "晚上去散步",
    "编写一个Python脚本", "开发一个新的应用程序", "编程任务有点多", "学习Java编程", "写一个新的博客网站",
    "计划去爬山", "准备去郊区露营", "周末去海边玩", "下个月去国外旅游", "计划去森林里探险",
    "和男朋友去约会", "和女朋友去看展览", "一起吃晚餐", "和女朋友去游乐园玩", "和男朋友去逛街",
    "写一篇关于机器学习的论文", "准备写一篇自然语言处理的论文", "写一篇关于数据挖掘的论文", "写一篇关于计算机图形学的论文", "准备写一篇深度学习的论文"
]

labels += [
    "吃饭", "吃饭", "吃饭", "吃饭", "吃饭",
    "睡觉", "睡觉", "睡觉", "睡觉", "睡觉",
    "写作业", "写作业", "写作业", "写作业", "写作业",
    "锻炼", "锻炼", "锻炼", "锻炼", "锻炼",
    "编程", "编程", "编程", "编程", "编程",
    "出游", "出游", "出游", "出游", "出游",
    "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱",
    "写论文", "写论文", "写论文", "写论文", "写论文"
]

texts += [
    "晚饭想吃烧烤", "午餐吃点什么", "找一家好吃的饭馆", "晚上吃什么好呢", "午饭吃什么好",
    "今天想早睡", "昨晚做了很多梦", "需要补个午觉", "今天早点睡", "昨晚失眠了",
    "历史作业还没写完", "英语作业很多", "作业太多写不完", "今天要完成所有作业", "物理作业太难了",
    "下午去打篮球", "早上晨跑很舒服", "去健身房锻炼", "晚上去散步", "下午去游泳",
    "编写一个新的程序", "正在学习Java编程", "编写一个新的应用", "学习数据结构", "写一个自动化脚本",
    "准备去沙滩度假", "打算去乡下玩", "计划去野营", "下周去爬山", "下个月去旅行",
    "和女朋友去约会", "和男朋友去逛街", "一起看电影", "和女朋友去游乐园", "和男朋友一起吃晚餐",
    "写一篇关于深度学习的论文", "写一篇机器翻译的论文", "准备写一篇大数据论文", "写一篇关于自然语言处理的论文", "写一篇关于计算机视觉的论文"
]

labels += [
    "吃饭", "吃饭", "吃饭", "吃饭", "吃饭",
    "睡觉", "睡觉", "睡觉", "睡觉", "睡觉",
    "写作业", "写作业", "写作业", "写作业", "写作业",
    "锻炼", "锻炼", "锻炼", "锻炼", "锻炼",
    "编程", "编程", "编程", "编程", "编程",
    "出游", "出游", "出游", "出游", "出游",
    "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱",
    "写论文", "写论文", "写论文", "写论文", "写论文"
]

texts += [
    "今天中午想吃快餐", "晚餐吃点中餐", "找一家好的餐厅吃晚饭", "午餐吃什么菜好呢", "晚上吃寿司怎么样",
    "今晚早点睡觉", "昨晚失眠了", "今天需要补觉", "晚上早点休息", "昨晚梦见了老同学",
    "化学作业还没做", "数学作业很多", "今天要赶完所有作业", "周末还有作业要写", "物理作业非常难",
    "今天去健身房锻炼", "早上跑步很舒服", "去做瑜伽", "下午去打网球", "晚上去散步",
    "编写一个新的程序", "开发一个新的应用", "编程任务太多了", "学习Python编程", "写一个新的博客",
    "计划去海边度假", "准备去山里露营", "周末去城市游玩", "下个月去国外旅游", "计划去森林里探险",
    "和男朋友去约会", "和女朋友去看展览", "一起吃晚餐", "和女朋友去游乐园玩", "和男朋友去逛街",
    "写一篇关于机器学习的论文", "准备写一篇大数据论文", "写一篇关于人工智能的论文", "写一篇关于区块链的论文", "准备写一篇深度学习的论文"
]

labels += [
    "吃饭", "吃饭", "吃饭", "吃饭", "吃饭",
    "睡觉", "睡觉", "睡觉", "睡觉", "睡觉",
    "写作业", "写作业", "写作业", "写作业", "写作业",
    "锻炼", "锻炼", "锻炼", "锻炼", "锻炼",
    "编程", "编程", "编程", "编程", "编程",
    "出游", "出游", "出游", "出游", "出游",
    "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱",
    "写论文", "写论文", "写论文", "写论文", "写论文"
]

texts += [
    "今天中午想吃面条", "晚餐吃点意大利面", "找一家好的餐厅吃午饭", "午餐吃什么好呢", "晚上吃披萨怎么样",
    "今晚早点睡觉", "昨晚睡得很香", "今天需要午睡一下", "晚上早一点休息", "昨晚梦见了朋友",
    "化学作业还没完成", "数学作业很多", "今天要赶完所有作业", "周末还有作业要做", "物理作业太多了",
    "今天去游泳", "早上跑步感觉很好", "去健身房锻炼身体", "下午去打篮球", "晚上去散步",
    "编写一个新的脚本", "开发一个新的应用程序", "编程任务太多", "学习Python编程", "写一个新的博客文章",
    "计划去海边玩", "准备去山里露营", "周末去城市观光", "下个月去国外旅游", "计划去森林探险",
    "和男朋友约会", "和女朋友去购物", "一起吃晚餐", "和女朋友去游乐场", "和男朋友一起看电影",
    "写一篇关于数据科学的论文", "准备写一篇机器学习的论文", "写一篇关于人工智能的论文", "写一篇关于区块链的论文", "准备写一篇深度学习的论文"
]

labels += [
    "吃饭", "吃饭", "吃饭", "吃饭", "吃饭",
    "睡觉", "睡觉", "睡觉", "睡觉", "睡觉",
    "写作业", "写作业", "写作业", "写作业", "写作业",
    "锻炼", "锻炼", "锻炼", "锻炼", "锻炼",
    "编程", "编程", "编程", "编程", "编程",
    "出游", "出游", "出游", "出游", "出游",
    "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱",
    "写论文", "写论文", "写论文", "写论文", "写论文"
]
texts += [
    "今晚吃火锅吧", "午餐吃点中式快餐", "找家饭馆吃晚餐", "午饭吃什么好呢", "晚上吃寿司如何",
    "今天晚上早点睡", "昨晚睡得很香", "午睡一下恢复精神", "今晚早点上床休息", "昨晚梦见了老同学",
    "化学作业还没写完", "数学作业堆积如山", "今天晚上要赶作业", "周末还有作业要完成", "物理作业太难了",
    "今天下午去打羽毛球", "早上晨跑真舒服", "去健身房锻炼身体", "晚上去散步", "下午去游泳",
    "编写一个新的应用程序", "开发一个小游戏", "编程任务好多", "学习Java编程", "写一个自动化脚本",
    "计划去海边度假", "准备去山里露营", "周末去城市游玩", "下个月去国外旅游", "计划去森林里探险",
    "和男朋友去约会", "和女朋友逛街", "一起看电影", "和女朋友去游乐园玩", "和男朋友一起吃晚餐",
    "写一篇关于深度学习的论文", "写一篇机器翻译的论文", "准备写一篇大数据论文", "写一篇关于自然语言处理的论文", "写一篇关于计算机视觉的论文"
]

labels += [
    "吃饭", "吃饭", "吃饭", "吃饭", "吃饭",
    "睡觉", "睡觉", "睡觉", "睡觉", "睡觉",
    "写作业", "写作业", "写作业", "写作业", "写作业",
    "锻炼", "锻炼", "锻炼", "锻炼", "锻炼",
    "编程", "编程", "编程", "编程", "编程",
    "出游", "出游", "出游", "出游", "出游",
    "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱",
    "写论文", "写论文", "写论文", "写论文", "写论文"
]

texts += [
    "今天想吃韩式烤肉", "晚餐吃点什么好呢", "找一家好的餐厅吃午饭", "午餐吃什么菜好", "晚上吃披萨怎么样",
    "今晚早点休息", "昨晚梦见了家人", "今天需要午睡一下", "晚上早一点休息", "昨晚失眠了",
    "物理作业还没完成", "化学作业很多", "今天要赶完所有作业", "周末还有作业要做", "数学作业非常难",
    "今天去跑步", "早上晨跑感觉很好", "去健身房锻炼身体", "下午去打篮球", "晚上去散步",
    "编写一个新的程序", "开发一个新的应用程序", "编程任务太多", "学习Python编程", "写一个新的博客文章",
    "计划去海边玩", "准备去山里露营", "周末去城市观光", "下个月去国外旅游", "计划去森林探险",
    "和女朋友约会", "和男朋友去逛街", "一起看电影", "和女朋友去游乐园", "和男朋友一起吃晚餐",
    "写一篇关于机器学习的论文", "准备写一篇大数据论文", "写一篇关于人工智能的论文", "写一篇关于区块链的论文", "准备写一篇深度学习的论文"
]

labels += [
    "吃饭", "吃饭", "吃饭", "吃饭", "吃饭",
    "睡觉", "睡觉", "睡觉", "睡觉", "睡觉",
    "写作业", "写作业", "写作业", "写作业", "写作业",
    "锻炼", "锻炼", "锻炼", "锻炼", "锻炼",
    "编程", "编程", "编程", "编程", "编程",
    "出游", "出游", "出游", "出游", "出游",
    "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱", "谈恋爱",
    "写论文", "写论文", "写论文", "写论文", "写论文"
]



def jieba_tokenizer(text):
    # 使用 OpenCC 将繁体转换为简体
    converter = opencc.OpenCC('t2s')
    text = converter.convert(text)
    
    # 使用正则表达式去除所有字母、数字以及非中文字符，只保留中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    
    # 使用 jieba 进行分词
    tokens = jieba.lcut(text)
    
    # 移除停用词
    tokens = [token for token in tokens if token not in stopwords]
    
    return ' '.join(tokens)

# 确保标签数据一致性
def validate_labels(labels, unique_labels):
    valid_labels = [label for label in labels if label in unique_labels]
    return valid_labels

# 处理标签一致性
labels = validate_labels(labels, unique_labels)

# 检查标签数量是否正确
assert len(set(labels)) == len(unique_labels), "标签数量不一致，请检查标签数据"

# 处理文本数据
tokenized_texts = [jieba_tokenizer(text) for text in texts]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(tokenized_texts, labels, test_size=0.2, random_state=42)

# 创建并训练模型
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 输出分类测试正确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 输出混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
sns.heatmap(cm, annot=False, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, cmap='Blues')
#plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 计算多分类问题的 ROC 和 AUC
y_test_bin = label_binarize(y_test, classes=unique_labels)
y_prob = model.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i, label in enumerate(unique_labels):
    fpr[label], tpr[label], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[label] = auc(fpr[label], tpr[label])

plt.figure()
for label in unique_labels:
    plt.plot(fpr[label], tpr[label], label=f'ROC curve of class {label} (area = {roc_auc[label]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision = dict()
recall = dict()
for i, label in enumerate(unique_labels):
    precision[label], recall[label], _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(recall[label], precision[label], label=f'Precision-Recall curve of class {label}')
plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# 示例预测
new_texts = [
    "焦虑的只想睡觉", "我写论文写的想死", "我是一个程序猿"
]
tokenized_new_texts = [jieba_tokenizer(text) for text in new_texts]
new_predictions = model.predict(tokenized_new_texts)

# 打印预测结果表格
print("新文本预测结果：")
for text, prediction in zip(new_texts, new_predictions):
    print(f"Text: {text} => Intent: {prediction}")