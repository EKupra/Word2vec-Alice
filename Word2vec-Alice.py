import numpy as np
import random
import re
from collections import Counter

random.seed(0)
np.random.seed(0)

try:
    with open("Alice.txt", "r", encoding="utf-8") as f:
        text = f.read()
except UnicodeDecodeError:
    with open("Alice.txt", "r", encoding="latin-1") as f:
        text = f.read()

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
end_marker   = "*** END OF THE PROJECT GUTENBERG EBOOK"

start = text.find(start_marker)
end = text.find(end_marker)

if start != -1 and end != -1:
    text = text[start:end]

text = text.lower()
text = re.sub(r"[^a-z\s]", " ", text)
words = text.split()

roman = {"i","ii","iii","iv","v","vi","vii","viii","ix","x"}
words = [w for w in words if w not in roman]

word_counts = Counter(words)

min_freq = 5
vocab = [w for w, c in word_counts.items() if c >= min_freq]

word_to_idx = {}
for i, w in enumerate(vocab):
    word_to_idx[w] = i

idx_to_word = {i: w for w, i in word_to_idx.items()}

indexed_words = [word_to_idx[w] for w in words if w in word_to_idx]

window_size = 3
training_data = []

for i in range(len(indexed_words)):
    target = indexed_words[i]
    start = max(0, i - window_size)
    end = min(len(indexed_words), i + window_size + 1)

    context = []
    for j in range(start, end):
        if j != i:
            context.append(indexed_words[j])

    if len(context) > 0:
        training_data.append((context, target))

V = len(vocab)
D = 100

W_in = np.random.randn(V, D) * 0.01
W_out = np.random.randn(V, D) * 0.01

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))

counts = np.array([word_counts[w] for w in vocab], dtype=np.float64)
neg_sampling_probs = counts ** 0.75
neg_sampling_probs /= neg_sampling_probs.sum()

def get_negative_samples(true_idx, K):
    negatives = set()
    while len(negatives) < K:
        sample = int(np.random.choice(V, p=neg_sampling_probs))
        if sample != true_idx:
            negatives.add(sample)
    return list(negatives)

def train_step(context_indices, target_idx, lr=0.005, K=5):
    context_vectors = W_in[context_indices]
    v = np.mean(context_vectors, axis=0)

    u_pos = W_out[target_idx]
    score_pos = np.dot(v, u_pos)
    prob_pos = sigmoid(score_pos)
    loss = -np.log(prob_pos + 1e-12)

    grad_v = (prob_pos - 1) * u_pos
    grad_u_pos = (prob_pos - 1) * v

    W_out[target_idx] -= lr * grad_u_pos

    negatives = get_negative_samples(target_idx, K)

    for neg_idx in negatives:
        u_neg = W_out[neg_idx]
        score_neg = np.dot(v, u_neg)
        prob_neg = sigmoid(score_neg)
        loss += -np.log(1 - prob_neg + 1e-12)

        grad_v += prob_neg * u_neg
        grad_u_neg = prob_neg * v

        W_out[neg_idx] -= lr * grad_u_neg

    grad_context = grad_v / len(context_indices)

    for ctx_idx in context_indices:
        W_in[ctx_idx] -= lr * grad_context

    return loss

epochs = 5
lr = 0.005
K = 10

for epoch in range(epochs):
    random.shuffle(training_data)
    total_loss = 0

    for i, (context, target) in enumerate(training_data):
        total_loss += train_step(context, target, lr, K)

        if (i + 1) % 20000 == 0:
            print("Epoch", epoch + 1, "step", i + 1, "avg loss", total_loss / (i + 1))

    print("End epoch", epoch + 1, "avg loss", total_loss / len(training_data))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def most_similar(word, top_k=10):
    if word not in word_to_idx:
        print("Word not in vocab")
        return

    w_idx = word_to_idx[word]
    w_vec = W_in[w_idx]

    sims = []
    for i in range(V):
        if i == w_idx:
            continue
        sim = cosine_similarity(w_vec, W_in[i])
        sims.append((sim, i))

    sims.sort(reverse=True)

    print("Most similar to", word)
    for sim, idx in sims[:top_k]:
        print(idx_to_word[idx], sim)

stopwords = {
    "a", "an", "the",
    "and", "or", "but",
    "to", "of", "in", "on", "at", "for", "from", "with", "by", "about",
    "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our",
    "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those",
    "as", "not", "no", "so", "if", "then", "than",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "can", "could", "would", "should", "may", "might", "must", "will", "shall",
    "said", "say", "says",
    "one", "little", "much", "many", "some", "only", "very", "well", "like", "know",
    "up", "down", "out", "into", "again", "there", "here", "away", "off",
    "t", "m", "ll", "don", "its", "ve"
}

top_common = {w for w, _ in word_counts.most_common(20)}
stopwords = stopwords.union(top_common)

def most_similar_filtered(word, top_k=10):
    if word not in word_to_idx:
        print("Word not in vocab")
        return

    w_idx = word_to_idx[word]
    w_vec = W_in[w_idx]

    sims = []
    for i in range(V):
        if i == w_idx:
            continue

        candidate = idx_to_word[i]
        if candidate in stopwords:
            continue

        sim = cosine_similarity(w_vec, W_in[i])
        sims.append((sim, i))

    sims.sort(reverse=True)

    print("Most similar to", word, "(filtered)")
    for sim, idx in sims[:top_k]:
        print(idx_to_word[idx], sim)

most_similar("alice")
print()
most_similar_filtered("alice")
print("\n")

most_similar("rabbit")
print()
most_similar_filtered("rabbit")
print("\n")

most_similar("queen")
print()
most_similar_filtered("queen")