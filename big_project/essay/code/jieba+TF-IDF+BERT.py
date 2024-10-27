import jieba

# 示例文本
texts = ["这是一个测试文本。我们需要进行分词处理。", "这是另一个测试文本。"]

# 使用 jieba 进行分词
segmented_texts = [" ".join(jieba.cut(text)) for text in texts]
from sklearn.feature_extraction.text import TfidfVectorizer

# 初始化 TfidfVectorizer
vectorizer = TfidfVectorizer()

# 拟合数据并转换为特征向量
X_tfidf = vectorizer.fit_transform(segmented_texts)

# 获取词汇表
vocab = vectorizer.get_feature_names_out()
print(vocab)

# 特征向量表示
print(X_tfidf.toarray())
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# 初始化 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 获取 BERT 嵌入
bert_embeddings = [get_bert_embedding(text) for text in texts]

# 打印 BERT 嵌入的形状
print([embedding.shape for embedding in bert_embeddings])
# 分别对TF-IDF和BERT特征向量进行处理
tfidf_sum = np.sum(X_tfidf.toarray(), axis=0)
bert_sum = np.sum(np.vstack(bert_embeddings), axis=0)

# 选择TF-IDF的前10个热点词
tfidf_hotspot_indices = np.argsort(tfidf_sum)[0:]
tfidf_hotspot_words = [vocab[i] for i in tfidf_hotspot_indices]
print("TF-IDF Hotspot words:", tfidf_hotspot_words)

# BERT特征处理（需要进一步处理以映射到词汇表）
# 这里假设我们已经有一个将BERT嵌入映射回词汇表的函数或步骤
# 例如，可以计算每个词的BERT嵌入，并找出与bert_sum最接近的词
# 这是一个示例，具体实现可以根据需求调整
def find_closest_words(bert_embedding, vocab, model, tokenizer):
    word_embeddings = []
    for word in vocab:
        inputs = tokenizer(word, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        word_embeddings.append(word_embedding)
    
    word_embeddings = np.array(word_embeddings)
    similarities = np.dot(word_embeddings, bert_embedding) / (np.linalg.norm(word_embeddings, axis=1) * np.linalg.norm(bert_embedding))
    return [vocab[i] for i in np.argsort(similarities)[0:]]

bert_hotspot_words = find_closest_words(bert_sum, vocab, model, tokenizer)
print("BERT Hotspot words:", bert_hotspot_words)
