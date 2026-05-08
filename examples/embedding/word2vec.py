from typing import Optional
import numpy as np
from deeplearning.network import Network
from deeplearning.layers import Dense, ActivationLayer
from deeplearning.activations import softmax, softmax_prime
from deeplearning.losses import cross_entropy, cross_entropy_prime
import matplotlib.pyplot as plt

def neighbor(context: str, window_size=1):
    tokens = context.split()
    for i in range(len(tokens)):
        for window in range(1, window_size + 1):
            if i - window >= 0:
                yield (tokens[i], tokens[i - window])
            if i + window < len(tokens):
                yield (tokens[i], tokens[i + window])
                
class Word2VecTrainer:
    def __init__(self, context):
        self.context = context
        words = context.split()

        self.word_to_id = {}
        self.id_to_word = {}

        current_id = 1

        for word in words:
            if not word.strip():
                continue

            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1

    def word2id(self, word: str) -> int:
        return self.word_to_id.get(word, 0)

    def id2word(self, id: int) -> str:
        return self.id_to_word.get(id, "")

    def create_training_data(self, pairs):
        vocab_size = len(self.word_to_id)
        num_pairs = len(pairs)

        X = np.zeros((num_pairs, vocab_size))
        Y = np.zeros((num_pairs, vocab_size))

        for i, (target, context) in enumerate(pairs):
            target_id = self.word_to_id[target] - 1
            context_id = self.word_to_id[context] - 1

            X[i, target_id] = 1
            Y[i, context_id] = 1

        return X, Y

text = """
the king is a royal man
the queen is a royal woman
the boy is a young man
the girl is a young woman
the prince is a royal boy
the princess is a royal girl
man is to king as woman is to queen
"""

pairs = list(neighbor(text, window_size=1))

w2v = Word2VecTrainer(text)

X, Y = w2v.create_training_data(pairs)



vocab_size = len(w2v.word_to_id) 
embedding_size = 40

net = Network()
net.add(Dense(vocab_size, embedding_size)) 
net.add(Dense(embedding_size, vocab_size)) 
net.add(ActivationLayer(softmax, softmax_prime))

net.use(cross_entropy, cross_entropy_prime)

net.fit(X, Y, epochs=8000, learning_rate=0.01)


weights = net.layers[0].weights 

def get_vector(word):
    word_id = w2v.word2id(word) - 1
    s = weights[word_id]
    return weights[word_id]

king_vec = get_vector("king")
print("King vector:", king_vec)

man_vec = get_vector("man")
print("Man vector:", man_vec)

woman_vec = get_vector("woman")
print("Woman vector:", woman_vec)

target_vec = king_vec - man_vec + woman_vec
print("Target vector: " , target_vec)

queen_vec = get_vector("queen")
print("Queen vector:", queen_vec)


def plot_embeddings(weights, trainer, words_to_plot):
    plt.figure(figsize=(10, 8))
    
    for word in words_to_plot:
        word_vec = get_vector(word)
        
        x, y = word_vec[0], word_vec[1]
        
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.grid(True)
    plt.title("نمایش کلمات در فضای برداری (Embedding Space)")
    plt.show()

words = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]
plot_embeddings(net.layers[0].weights, w2v, words)