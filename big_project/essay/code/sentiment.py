from transformers import pipeline

# 加载预训练的情感分析模型
sentiment_analyzer = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    # 使用模型进行情感分析
    result = sentiment_analyzer(text)[0]
    return result

# 测试文本
texts = [
    "我今天很开心。",
    "这个产品太差劲了。",
    "服务态度非常好。",
    "这部电影让我很失望。",
    "天气真好，适合出去玩。",
    "我的电脑坏了，我很郁闷。",
    "这是我吃过最好吃的披萨。",
    "这家餐厅的服务让人无语。",
    "我对这次旅行非常满意。",
    "他总是迟到，真让人生气。"
]

# 分析结果
results = [analyze_sentiment(text) for text in texts]

# 打印结果
print(f"{'文本':<30} {'情感类别':<15} {'置信度':<10}")
print("="*60)
for text, result in zip(texts, results):
    print(f"{text:<30} {result['label']:<15} {result['score']:.4f}")
