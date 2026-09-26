# predict.py
import numpy as np
from load_data import load_dataset
from preprocess import preprocess_data
from train import train_model

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def predict_samples(model, x_test, y_test, num_samples=5):
    predictions = model.predict(x_test[:num_samples])
    for i in range(num_samples):
        pred_label = class_names[np.argmax(predictions[i])]
        true_label = class_names[np.argmax(y_test[i])]
        print(f"Sample {i}: Predicted = {pred_label}, Actual = {true_label}")

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset()
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    model, history = train_model(5, 2, 32, x_train, y_train, x_test, y_test)
    predict_samples(model, x_test, y_test)