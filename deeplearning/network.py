import numpy as np 

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        for layer in self.layers:
            layer.is_training = False
            
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate, batch_size=32):
        samples = len(x_train)
    
        for i in range(epochs):
            err = 0
            indices = np.arange(samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]
    
            for j in range(0, samples, batch_size):
                x_batch = x_train[j:j + batch_size]
                y_batch = y_train[j:j + batch_size]
    
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)
    
                err += self.loss(y_batch, output)
    
                error = self.loss_prime(y_batch, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
    
            err /= (samples / batch_size)
            print(f"Epoch {i+1}/{epochs}   |   Error: {err:.6f}")