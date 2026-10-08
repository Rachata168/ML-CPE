import tensorflow as tf

# โหลด MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Standardize ข้อมูล
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# เพิ่ม channel สำหรับ CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# ฟังก์ชันสร้าง DCNN
def create_model():
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


# ทดลองจำนวน Epoch ต่างกัน
epoch_values = [5, 10, 15]

for epochs in epoch_values:

    print("\n==============================")
    print(f"Training with {epochs} epochs")
    print("==============================")

    # สร้างโมเดลใหม่ทุกครั้ง
    model = create_model()

    # Train model
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # ประเมินผลกับ Test set
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")