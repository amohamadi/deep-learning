from layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()

        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        
        
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        
        return self.output
 
    def backward(self, output_error, learning_rate):
        batch_size = output_error.shape[0] # تعداد داده‌ها در این دسته
        
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)
    
        # آپدیت پارامترها (تقسیم بر batch_size برای پایداری)
        self.weights -= learning_rate * (weights_error / batch_size)
        self.bias -= learning_rate * (bias_error / batch_size)
    
        return input_error


