from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib

# 设置 Matplotlib 中文字体
matplotlib.rc("font", family="SimHei")  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese', output_attentions=True)

# Define sentences
sentences = [
    "我喜欢吃苹果",
    "我喜欢吃香蕉",
    "我不喜欢吃苹果",
    "他喜欢吃苹果",
    "我和他都喜欢吃苹果"
]

# Tokenize and prepare inputs
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
outputs = model(**encoded_inputs)

# Extract embeddings (last hidden state)
embeddings = outputs.last_hidden_state
# Average the token embeddings to get a single vector for each sentence
sentence_embeddings = torch.mean(embeddings, dim=1)

# Convert tensor to numpy for t-SNE
sentence_embeddings_np = sentence_embeddings.detach().numpy()

# Perform t-SNE to reduce dimensionality for visualization (2D)
# Set perplexity to a value less than the number of samples, here we can try 3 as a start.
tsne = TSNE(n_components=2, perplexity=3, learning_rate='auto', random_state=0)
sentence_embeddings_tsne = tsne.fit_transform(sentence_embeddings_np)

# Plotting with seaborn for better visuals
plt.figure(figsize=(12, 10))
sns.scatterplot(x=sentence_embeddings_tsne[:, 0], y=sentence_embeddings_tsne[:, 1], hue=sentences, palette=sns.color_palette("hsv", len(sentences)), s=100, legend='full')

# Adding annotations with some additional details
for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (sentence_embeddings_tsne[i, 0], sentence_embeddings_tsne[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')

#plt.title('BERT Sentence Embeddings Visualization (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.legend(title='Sentences', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# Set up the matplotlib figure and axes
fig, axs = plt.subplots(5, 1, figsize=(15, 25))  # One subplot per sentence

# Process each sentence and visualize the attention weights
for i, sentence in enumerate(sentences):
    # Encode and pass the sentence to the model
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    attentions = outputs.attentions  # Attention layers outputs

    # We take the mean of all attention heads in the last layer
    attention = attentions[-1].mean(dim=1)[0]  # Mean over all heads

    # Create a heatmap for the sentence's attention matrix
    sns.heatmap(attention.detach().numpy(), ax=axs[i], cmap='viridis', cbar=True,
                xticklabels=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
                yticklabels=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    axs[i].set_title(f"Sentence {i + 1}: {sentence}")

# Adjust layout
plt.tight_layout()
plt.show()
