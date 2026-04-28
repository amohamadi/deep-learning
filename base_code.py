import numpy as np

# ایمپورت کردن کلاس‌هایی که ساختی
from network import Network
from dense import Dense  # یا هر اسمی که روی فایل لایه خطی گذاشتی
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# ۱. آماده‌سازی داده‌های XOR
# ورودی‌ها: 4 تا حالت مختلف
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
# جواب‌های واقعی (خروجی‌ها)
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# ۲. ساخت شبکه
net = Network()

# ۳. چیدن لایه‌ها (معماری شبکه)
# لایه پنهان: 2 تا ورودی می‌گیره، می‌ده به 3 تا نورون
net.add(Dense(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))

# لایه خروجی: اون 3 تا خروجی رو می‌گیره و تبدیلش می‌کنه به 1 جواب نهایی
net.add(Dense(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# ۴. تنظیم قاضیِ شبکه
net.use(mse, mse_prime)

# ۵. شروع آموزش!
print("--- شروع آموزش ---")
# داده‌ها رو 1000 بار با سرعت 0.1 مرور می‌کنیم
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# ۶. تست کردن شبکه بعد از آموزش
print("\n--- تست کردن شبکه ---")
out = net.predict(x_train)

# چاپ نتایج به صورت خوانا
for i in range(len(x_train)):
    # ورودی واقعی
    inp = x_train[i][0]
    # پیش‌بینی شبکه (چون خروجی ماتریسه، اولین عددش رو می‌گیریم)
    pred = out[i][0][0]
    # جواب واقعی
    true_val = y_train[i][0][0]
    
    print(f"Input: {inp} | Predicted: {pred:.4f} | True Value: {true_val}")