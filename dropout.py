from layer import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, input_data):
        if not self.is_training:
            return input_data  # در حالت تست، Dropout را اعمال نمی‌کنیم
        # ساخت یک ماتریس از صفر و یک با همان ابعاد ورودی
        # np.random.binomial(n, p, size)
        # n=1 (توزیع برنولی)، p=شانس یک بودن
        self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape)
        
        # اعمال ماسک و بزرگ‌نمایی (Scale) برای حفظ میانگین
        return (input_data * self.mask) / (1 - self.rate)

    def backward(self, output_error, learning_rate):
        # در مسیر برگشت، فقط خطا را از همان مسیرهایی که در رفت باز بودند عبور می‌دهیم
        return (output_error * self.mask) / (1 - self.rate)