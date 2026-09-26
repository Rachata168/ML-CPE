# load_data.py
from tensorflow.keras.datasets import fashion_mnist

def load_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print("Train shape:", x_train.shape, y_train.shape)
    print("Test shape:", x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    load_dataset()