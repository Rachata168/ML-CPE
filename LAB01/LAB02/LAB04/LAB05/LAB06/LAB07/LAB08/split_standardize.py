import tensorflow as tf

# โหลด MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# แสดงจำนวนข้อมูลก่อน Standardize
print("Before Standardize")
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

# Standardize ข้อมูลให้อยู่ในช่วง 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# เพิ่ม channel สำหรับการใช้ CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# แสดงขนาดข้อมูลหลังเตรียมข้อมูล
print("\nAfter Standardize")
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

print("\nData preparation completed.")