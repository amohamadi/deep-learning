from network import Network
from dense import Dense
from dropout import Dropout
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, softmax, softmax_prime
from losses import cross_entropy, cross_entropy_prime
from mnist_loader import load_mnist, preprocess_data
import numpy as np

images, labels = load_mnist('/Users/alireza/Desktop/MNIST_Dataset', kind='train')
x_train, y_train = preprocess_data(images, labels)

# ۲. ساخت شبکه
net = Network()

# لایه اول: ۷۸۴ ورودی به ۱۲۸ نورون (لایه پنهان)
net.add(Dense(784, 128))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(Dropout(0.2))
# لایه دوم: ۱۲۸ ورودی به ۶۴ نورون (لایه پنهان دوم برای دقت بیشتر)
net.add(Dense(128, 64))
net.add(ActivationLayer(tanh, tanh_prime))

# لایه خروجی: ۶۴ ورودی به ۱۰ خروجی نهایی
net.add(Dense(64, 10))
net.add(ActivationLayer(softmax, softmax_prime))

# ۳. تنظیم تابع خطا
net.use(cross_entropy, cross_entropy_prime)

# ۴. آموزش (با Batch Size مثلاً ۳۲ یا ۶۴)
# نکته: برای شروع، تعداد Epochs را کم بگذار (مثلاً ۱۰) تا سرعت را ببینی
net.fit(x_train, y_train, epochs=10, learning_rate=0.1, batch_size=32)



# # ۱. بارگذاری داده‌های تست (که مدل تا حالا ندیده)
# images_test, labels_test = load_mnist('./MNIST_Dataset', kind='t10k')
# x_test, y_test = preprocess_data(images_test, labels_test)

# # ۲. انجام پیش‌بینی برای تمام داده‌های تست
# print("\n--- در حال تست روی ۱۰,۰۰۰ تصویر جدید ---")
# predictions = net.predict(x_test)

# # ۳. محاسبه دقت (Accuracy)
# correct_predictions = 0
# for i in range(len(x_test)):
#     # پیدا کردن ایندکسی که بیشترین احتمال را دارد (عددی که شبکه حدس زده)
#     predicted_digit = np.argmax(predictions[i])
#     # پیدا کردن عدد واقعی از روی One-Hot
#     true_digit = np.argmax(y_test[i])
    
#     if predicted_digit == true_digit:
#         correct_predictions += 1

# accuracy = (correct_predictions / len(x_test)) * 100
# print(f"Accuracy: {accuracy:.2f}%")