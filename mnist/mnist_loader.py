import numpy as np
import struct

def load_mnist(path, kind='train'):
    """بارگذاری داده‌های MNIST از مسیر مشخص شده"""
    labels_path = f"{path}/{kind}-labels-idx1-ubyte/{kind}-labels-idx1-ubyte"
    images_path = f"{path}/{kind}-images-idx3-ubyte/{kind}-images-idx3-ubyte"
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels

def preprocess_data(images, labels):
    # نرمال‌سازی
    images = images.astype('float32') / 255
    
    # تبدیل برچسب‌ها به One-Hot
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels] = 1
    
    # --- تغییر مهم ---
    # به جای (N, 1, 784) بگذار فقط (N, 784) باشد
    images = images.reshape(images.shape[0], 784)
    # به جای (N, 1, 10) بگذار فقط (N, 10) باشد
    one_hot_labels = one_hot_labels.reshape(one_hot_labels.shape[0], 10)
    
    return images, one_hot_labels