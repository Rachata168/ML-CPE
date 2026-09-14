# โปรแกรม SVM Classification สำหรับ Iris Dataset
# ขั้นตอนหลัก: โหลดข้อมูล -> แบ่ง Train/Test -> ปรับมาตรฐาน
# -> ทดลอง 3 Kernel -> วัดผล -> เลือก Kernel ที่ดีที่สุด -> ทำนาย
#
# ใช้จัดการข้อมูลในรูปแบบตาราง
import pandas as pd
# ใช้คำนวณและจัดการข้อมูลตัวเลข
import numpy as np
# ใช้สร้างกราฟ
import matplotlib.pyplot as plt
# ใช้สร้างกราฟและ Heatmap
import seaborn as sns
# โหลดชุดข้อมูล Iris
from sklearn.datasets import load_iris
# แบ่งข้อมูล Train และ Test
from sklearn.model_selection import train_test_split
# ปรับมาตรฐานข้อมูล
from sklearn.preprocessing import StandardScaler
# เรียกใช้โมเดล SVM
from sklearn.svm import SVC
# เครื่องมือวัดผลโมเดล
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# โหลด Iris Dataset
iris = load_iris()
# X คือข้อมูลคุณลักษณะของดอกไม้
X = iris.data
# y คือประเภทของดอกไม้
y = iris.target
# ชื่อคุณลักษณะของข้อมูล
feature_names = iris.feature_names
# ชื่อประเภทของดอกไม้
target_names = iris.target_names

# เปลี่ยนข้อมูลเป็นตาราง
df = pd.DataFrame(X, columns=feature_names)
# เพิ่มประเภทดอกไม้ลงในตาราง
df['target'] = y
print(df.head())
print(df.describe())
print(df['target'].value_counts())

# สร้างกราฟดูความสัมพันธ์ของข้อมูล
sns.pairplot(df, hue='target')
plt.savefig('pairplot.png')
plt.close()

# แบ่งข้อมูล 80% Train และ 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# สร้างตัวปรับมาตรฐานข้อมูล
scaler = StandardScaler()
# ปรับมาตรฐานข้อมูล Train
X_train_scaled = scaler.fit_transform(X_train)
# ปรับมาตรฐานข้อมูล Test
X_test_scaled = scaler.transform(X_test)

# Kernel ที่ต้องการทดลอง 3 แบบ
kernels = ['linear', 'poly', 'rbf']
# เก็บค่า Accuracy ของแต่ละ Kernel
results = {}

# ทดลองสร้างโมเดลทีละ Kernel
for kernel in kernels:
    # สร้างโมเดล SVM
    model = SVC(kernel=kernel, random_state=42)
    # ฝึกโมเดลด้วยข้อมูล Train
    model.fit(X_train_scaled, y_train)
    # ทำนายข้อมูล Test
    y_pred = model.predict(X_test_scaled)
    # คำนวณ Accuracy
    acc = accuracy_score(y_test, y_pred)
    # เก็บ Accuracy ไว้เปรียบเทียบ
    results[kernel] = acc

    print(f"\n=== Kernel: {kernel} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # สร้าง Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {kernel}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{kernel}.png')
    plt.close()

# แสดงผลสรุป Accuracy ของแต่ละ Kernel
print("\n=== Summary ===")
# แสดง Accuracy ของแต่ละ Kernel
for kernel, acc in results.items():
    print(f"{kernel}: {acc:.4f}")

plt.figure()
# สร้างกราฟเปรียบเทียบ Accuracy
plt.bar(results.keys(), results.values(), color=['skyblue', 'orange', 'green'])
plt.ylabel('Accuracy')
plt.title('SVM Kernel Comparison')
plt.ylim(0, 1)
plt.savefig('kernel_comparison.png')
plt.close()

# เลือกข้อมูลตัวอย่าง 5 รายการ
sample = X_test_scaled[:5]
# เลือก Kernel ที่ Accuracy สูงที่สุด
best_kernel = max(results, key=results.get)
# สร้างโมเดลจาก Kernel ที่ดีที่สุด
best_model = SVC(kernel=best_kernel, random_state=42)
# ฝึกโมเดลที่ดีที่สุด
best_model.fit(X_train_scaled, y_train)
# ทำนายข้อมูลตัวอย่าง 5 รายการ
predictions = best_model.predict(sample)
print(f"\nPredictions using best kernel ({best_kernel}):")
# แสดงผล Predicted เทียบกับ Actual
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Predicted = {target_names[pred]}, Actual = {target_names[y_test[i]]}")