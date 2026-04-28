import numpy as np

def mse(y_true, y_pred):
    """
    محاسبه میزان خطا. 
    از این برای چاپ کردن خطا در حین آموزش استفاده می‌کنیم تا ببینیم 
    آیا شبکه داره یاد می‌گیره یا نه.
    """
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """
    مشتق تابع خطا.
    این تابع، نقطه شروعِ مرحله backward را به ما می‌دهد.
    """
    # y_true.size همان n (تعداد داده‌ها) در فرمول است
    return 2 * (y_pred - y_true) / y_true.size


def cross_entropy(y_true, y_pred):
    # جلوگیری از خطای log(0)
    y_pred = np.clip(y_pred, 1e-15, 1. - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def cross_entropy_prime(y_true, y_pred):
    # مشتق این تابع وقتی با Softmax ترکیب شود بسیار ساده می‌شود
    # اما به تنهایی به این شکل است:
    return y_pred - y_true