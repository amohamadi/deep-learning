from ..layer import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, input_data):
        if not self.is_training:
            return input_data  
        self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape)
        
        return (input_data * self.mask) / (1 - self.rate)

    def backward(self, output_error, learning_rate):
        return (output_error * self.mask) / (1 - self.rate)