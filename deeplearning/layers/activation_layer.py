import numpy as np
from ..layer import Layer

class ActivationLayer(Layer):
    def __init__(self, activation_function, activation_prime):
        super().__init__()
        self.activation = activation_function
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = self.activation_prime(self.input) * output_error
        
        return input_error
    