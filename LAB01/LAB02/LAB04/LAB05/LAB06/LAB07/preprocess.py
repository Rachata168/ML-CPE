# preprocess.py
import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_data(x_train, y_train, x_test, y_test, num_classes=10):
    # Standardize: scale pixel values 0-255 -> 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to (samples, height, width, channels) for CNN
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    return x_train, y_train_cat, x_test, y_test_cat