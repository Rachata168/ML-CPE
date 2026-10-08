import tensorflow as tf

# โหลด MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Standardize ข้อมูล
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# เพิ่ม channel สำหรับ CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# สร้างโมเดลแบบที่ 1
def create_model_1():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            32, activation="relu"
        ),

        tf.keras.layers.Dense(
            10, activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# สร้างโมเดลแบบที่ 2
def create_model_2():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            64, activation="relu"
        ),

        tf.keras.layers.Dense(
            10, activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# เก็บโมเดลที่ต้องการเปรียบเทียบ
models = {
    "Model 1: 1 Conv Layer, 32 Neurons": create_model_1(),
    "Model 2: 2 Conv Layers, 64 Neurons": create_model_2()
}


# ฝึกและประเมินแต่ละโมเดล
for name, model in models.items():

    print("\n==============================")
    print(name)
    print("==============================")

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")