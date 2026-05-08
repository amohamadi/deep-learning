import numpy as np

embeddings = np.array([
    [1, 0, 1, 0], # AI
    [0, 2, 0, 1], # is
    [1, 1, 1, 1]  # amazing
])

np.random.seed(42)
W_q = np.random.randn(4, 4)
W_k = np.random.randn(4, 4)
W_v = np.random.randn(4, 4)

Q = np.dot(embeddings, W_q)
K = np.dot(embeddings, W_k)
V = np.dot(embeddings, W_v)

scores = np.dot(Q, K.T)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)

output = np.dot(attention_weights, V)

print("--- Attention Weights (Who looks at whom) ---")
print(attention_weights)
print("\n--- Final Output Shape ---")
print(output.shape)