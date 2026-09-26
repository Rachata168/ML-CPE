# build_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_cnn(num_conv_layers=2, num_filters=32, num_classes=10):
    model = Sequential()
    model.add(Conv2D(num_filters, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))

    # เพิ่ม convolutional layer ตามจำนวนที่กำหนด (สำหรับเปรียบเทียบ config)
    for _ in range(num_conv_layers - 1):
        model.add(Conv2D(num_filters * 2, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model