# train.py
from load_data import load_dataset
from preprocess import preprocess_data
from build_model import build_cnn

def train_model(epochs, num_conv_layers, num_filters, x_train, y_train, x_test, y_test):
    model = build_cnn(num_conv_layers=num_conv_layers, num_filters=num_filters)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=64,
        verbose=1
    )
    return model, history

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset()
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)

    # ตัวอย่าง: เทรน 1 config ไว้ทดสอบว่าโค้ดรันได้
    model, history = train_model(
        epochs=5, num_conv_layers=2, num_filters=32,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )