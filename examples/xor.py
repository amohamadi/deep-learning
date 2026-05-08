import numpy as np

from deeplearning.network import Network
from deeplearning.layers import Dense, ActivationLayer
from deeplearning.activations import tanh, tanh_prime
from deeplearning.losses import mse, mse_prime

x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()

net.add(Dense(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))

net.add(Dense(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)

print("--- Fit! ---")
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

print("\n--- Test network ---")
out = net.predict(x_train)

for i in range(len(x_train)):
    inp = x_train[i][0]
    pred = out[i][0][0]
    true_val = y_train[i][0][0]
    
    print(f"Input: {inp} | Predicted: {pred:.4f} | True Value: {true_val}")