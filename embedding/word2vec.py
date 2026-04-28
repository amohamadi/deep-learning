from typing import Optional
import numpy as np
from network import Network
from dense import Dense
from activation_layer import ActivationLayer
from activations import softmax, softmax_prime
from losses import cross_entropy, cross_entropy_prime
import matplotlib.pyplot as plt

def neighbor(context: str, window_size=1):
    tokens = context.split()
    for i in range(len(tokens)):
        for window in range(1, window_size + 1):
            if i - window >= 0:
                yield (tokens[i], tokens[i - window])
            if i + window < len(tokens):
                yield (tokens[i], tokens[i + window])
                
class Word2VecTrainer:
    def __init__(self, context):
        self.context = context
        words = context.split()

        self.word_to_id = {}
        self.id_to_word = {}

        current_id = 1

        for word in words:
            if not word.strip():
                continue

            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1

    def word2id(self, word: str) -> int:
        return self.word_to_id.get(word, 0)

    def id2word(self, id: int) -> str:
        return self.id_to_word.get(id, "")

    def create_training_data(self, pairs):
        # تعداد کل کلمات منحصر‌به‌فرد
        vocab_size = len(self.word_to_id)
        # تعداد کل جفت‌های آموزشی
        num_pairs = len(pairs)

        # ساخت دو ماتریس تمام صفر
        # ابعاد: (تعداد جفت‌ها، تعداد کل کلمات)
        X = np.zeros((num_pairs, vocab_size))
        Y = np.zeros((num_pairs, vocab_size))

        for i, (target, context) in enumerate(pairs):
            # پیدا کردن ID کلمات (منهای یک می‌کنیم چون IDهای تو از ۱ شروع شده ولی ایندکس آرایه از ۰)
            target_id = self.word_to_id[target] - 1
            context_id = self.word_to_id[context] - 1

            # یک کردنِ خانه مربوط به آن کلمه
            X[i, target_id] = 1
            Y[i, context_id] = 1

        return X, Y

text = """
the king is a royal man
the queen is a royal woman
the boy is a young man
the girl is a young woman
the prince is a royal boy
the princess is a royal girl
man is to king as woman is to queen
"""

pairs = list(neighbor(text, window_size=1))

w2v = Word2VecTrainer(text)

X, Y = w2v.create_training_data(pairs)



# تعداد کلمات منحصربه‌فرد تو
vocab_size = len(w2v.word_to_id) 
# اندازه برداری که دوست داری برای هر کلمه داشته باشی
embedding_size = 40

net = Network()
net.add(Dense(vocab_size, embedding_size)) # لایه Embedding (بدون اکتیویشن)
net.add(Dense(embedding_size, vocab_size)) # لایه پیش‌بینی همسایه
net.add(ActivationLayer(softmax, softmax_prime))

net.use(cross_entropy, cross_entropy_prime)

# حالا آموزش بده (X و Y همونایی هستن که ساختی)
net.fit(X, Y, epochs=8000, learning_rate=0.01)


# ۱. استخراج ماتریس وزن‌ها (هر سطر یک کلمه است)
# نکته: در کلاس Dense تو احتمالا سطرها نماینده کلمات هستند
weights = net.layers[0].weights 

# ۲. تابعی برای گرفتن وکتور هر کلمه
def get_vector(word):
    word_id = w2v.word2id(word) - 1
    s = weights[word_id]
    return weights[word_id]

# ۳. انجام عملیات ریاضی مشهور
king_vec = get_vector("king")
print("King vector:", king_vec)

man_vec = get_vector("man")
print("Man vector:", man_vec)

woman_vec = get_vector("woman")
print("Woman vector:", woman_vec)

# محاسبه وکتور هدف
target_vec = king_vec - man_vec + woman_vec
print("Target vector: " , target_vec)

# ۴. حالا باید ببینیم این target_vec به کدام کلمه نزدیک‌تر است؟
queen_vec = get_vector("queen")
print("Queen vector:", queen_vec)


def plot_embeddings(weights, trainer, words_to_plot):
    plt.figure(figsize=(10, 8))
    
    for word in words_to_plot:
        # ۱. گرفتن وکتور کلمه
        word_vec = get_vector(word)
        
        # ۲. استفاده از دو بعد اول برای رسم روی نمودار (x, y)
        x, y = word_vec[0], word_vec[1]
        
        # ۳. رسم نقطه و نوشتن نام کلمه کنار آن
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.grid(True)
    plt.title("نمایش کلمات در فضای برداری (Embedding Space)")
    plt.show()

# کلماتی که می‌خواهیم روی نمودار ببینیم
words = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]
plot_embeddings(net.layers[0].weights, w2v, words)