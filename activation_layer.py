import numpy as np
from layer import Layer 

class ActivationLayer(Layer):
    def __init__(self, activation_function, activation_prime):
        super().__init__()
        # تابع اصلی (برای مسیر رفت)
        self.activation = activation_function
        # مشتق تابع (برای مسیر برگشت)
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        # اعمال تابع ریاضی روی تمام عناصر ماتریس ورودی
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        # این لایه پارامتری برای آپدیت کردن ندارد (وزن و بایاس ندارد)
        # فقط باید خطا را محاسبه کرده و به لایه قبلی پاس دهد.
        
        # نکته بسیار مهم: اینجا ضرب ماتریسی (dot) نداریم!
        # اینجا ضرب نظیر به نظیر (element-wise) انجام می‌دهیم.
        input_error = self.activation_prime(self.input) * output_error
        
        return input_error
    