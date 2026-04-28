import numpy as np 

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # ۱. متد اضافه کردن لایه‌ها به شبکه
    def add(self, layer):
        self.layers.append(layer)

    # ۲. متد تنظیم کردن تابع خطا (قاضیِ شبکه)
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # ۳. متد پیش‌بینی (فقط مسیر رفت - برای تست کردن بعد از آموزش)
    def predict(self, input_data):
        for layer in self.layers:
            layer.is_training = False
            
        # تعداد داده‌ها رو می‌گیریم
        samples = len(input_data)
        result = []

        # داده‌ها رو یکی یکی به شبکه می‌دیم
        for i in range(samples):
            # خروجیِ هر لایه، میشه ورودیِ لایه بعدی
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate, batch_size=32):
        samples = len(x_train)
    
        for i in range(epochs):
            err = 0
            # داده‌ها را قبل از هر اپوک بر می‌زنیم (Shuffle) تا شبکه حفظ نکند
            indices = np.arange(samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]
    
            # حرکت روی داده‌ها با گام‌های اندازه Batch
            for j in range(0, samples, batch_size):
                # جدا کردن یک دسته (Batch)
                x_batch = x_train[j:j + batch_size]
                y_batch = y_train[j:j + batch_size]
    
                # --- مسیر رفت (Forward) ---
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)
    
                # محاسبه خطا برای چاپ
                err += self.loss(y_batch, output)
    
                # --- مسیر برگشت (Backward) ---
                error = self.loss_prime(y_batch, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
    
            err /= (samples / batch_size)
            print(f"Epoch {i+1}/{epochs}   |   Error: {err:.6f}")