import numpy as np

# تابع اصلی Tanh
def tanh(x):
    return np.tanh(x)

# مشتق تابع Tanh (برای مسیر برگشت)
def tanh_prime(x):
    # فرمول ریاضی مشتق تانژانت هیپربولیک: 1 - tanh(x)^2
    return 1 - np.tanh(x)**2

def softmax(x):
    # کم کردن ماکزیمم از سرریز شدن (Overflow) جلوگیری می‌کند
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_prime(x):
    # یک نکته مهم: در عمل وقتی Softmax و Cross-Entropy با هم استفاده می‌شوند،
    # مشتق ترکیبی آن‌ها بسیار ساده می‌شود. 
    # فعلاً برای اینکه ساختار لایه‌هایت حفظ شود، این را خنثی نگه می‌داریم
    # چون محاسبات اصلی را در Cross-Entropy Prime انجام می‌دهیم.
    return 1