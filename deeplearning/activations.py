import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_prime(x):
    return 1