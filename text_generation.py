import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample expanded data (you can also read from file for better results)
data = """Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. It has become one of the most transformative technologies of the 21st century. AI is used in various industries such as healthcare, finance, transportation, education, and entertainment. In healthcare, AI helps diagnose diseases, recommend treatments, and even assist in surgeries. In finance, it detects fraud and predicts market trends.

Machine Learning (ML) is a subset of AI that focuses on enabling machines to learn from data without being explicitly programmed. Supervised, unsupervised, and reinforcement learning are the main types of ML. ML is used in email filtering, recommendation systems, customer segmentation, and many other applications.

Deep Learning (DL) is a further subset of ML that uses artificial neural networks with many layers (hence "deep"). It is particularly good at handling large amounts of unstructured data such as images, audio, and text. DL has led to breakthroughs in fields like image classification, speech recognition, and natural language processing (NLP).

Natural Language Processing enables machines to understand, interpret, and generate human language. Applications include chatbots, translation systems, sentiment analysis, and voice assistants. Tools like GPT and BERT have made enormous progress in this area, enabling more human-like interactions.

Recurrent Neural Networks (RNNs), and specifically Long Short-Term Memory (LSTM) networks, are useful for sequence modeling — such as time series forecasting, language modeling, and text generation. LSTMs can remember information over long periods, which makes them ideal for tasks like writing summaries or predicting the next word in a sentence.

AI models are trained on large datasets. The quality and size of the data significantly affect the model’s performance. Training deep learning models requires high computational power, often using GPUs or TPUs. Frameworks like TensorFlow and PyTorch make it easier to design and train these models.

The future of AI holds even more promise. As models become more capable and data becomes more available, AI will integrate more deeply into our daily lives. Ethical considerations like bias, transparency, and privacy are also becoming more important in the development of AI systems.

AI is not just a technical field. It intersects with philosophy, ethics, economics, and public policy. Understanding these dimensions is crucial for ensuring AI serves humanity positively. Educational institutions are now offering specialized AI programs, and companies are investing heavily in AI research and development.

Autonomous vehicles, smart assistants, automated medical diagnostics, and intelligent robots are just a few examples of what AI can do. With responsible development, the future of AI can be bright, equitable, and full of opportunities."""



# Preprocess
corpus = data.lower().split(".")
corpus = [line.strip() for line in corpus if line.strip()]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Generate sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

# Prepare input/output
X = input_sequences[:, :-1]
y = to_categorical(input_sequences[:, -1], num_classes=total_words)

# Build model
model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_seq_len - 1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=100, verbose=0)

# Top-K sampling
def sample_with_top_k(preds, k=10, banned=[]):
    preds = np.asarray(preds).astype('float64')
    top_k_indices = preds.argsort()[-k:][::-1]
    
    # Filter out banned words
    filtered_indices = [i for i in top_k_indices if i not in banned]
    if not filtered_indices:
        filtered_indices = top_k_indices  # fallback
    
    top_k_probs = preds[filtered_indices]
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    return np.random.choice(filtered_indices, p=top_k_probs)

# Final anti-repetition text generator
def generate_text(seed_text, next_words=20, k=10):
    recent_words = []
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]

        banned_indices = [tokenizer.word_index.get(w, -1) for w in recent_words[-3:]]
        predicted_index = sample_with_top_k(preds, k=k, banned=banned_indices)

        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        seed_text += ' ' + output_word
        recent_words.append(output_word)

    return seed_text

# Test it
print(generate_text("artificial intelligence", 30))
print(generate_text("deep learning", 30))
