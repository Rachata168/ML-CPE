import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 1. โหลด dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. สำรวจข้อมูล
print("===== ข้อมูลตัวอย่าง =====")
print(X.head())
print("\n===== ข้อมูลทั่วไป =====")
print(X.info())
print("\n===== สถิติเบื้องต้น =====")
print(X.describe())
print("\n===== จำนวนแต่ละ class =====")
print(y.value_counts())

# 3. แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. เทรนและประเมินผลด้วย k หลายค่า
k_values = [3, 5, 7]
results = {}

print("\n===== ผลลัพธ์ KNN =====")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[k] = acc
    print(f"k = {k} : Accuracy = {acc:.4f}")

# 6. หาค่า k ที่ดีที่สุด
best_k = max(results, key=results.get)
print(f"\nค่า k ที่ดีที่สุดคือ k = {best_k} ด้วยความแม่นยำ {results[best_k]:.4f}")

# 7. สรุปผลการทดลอง
print("\n===== สรุปผลการทดลอง =====")
print(f"""
จากการทดลองใช้ KNN กับ Iris dataset โดยเปรียบเทียบค่า k = 3, 5, 7 พบว่า
k = {best_k} ให้ค่าความแม่นยำสูงที่สุดที่ {results[best_k]:.4f}
เนื่องจาก Iris dataset มีขนาดเล็กและ class ค่อนข้างแยกกันชัดเจน
การเลือกค่า k ที่เหมาะสมจึงช่วยลดผลกระทบจาก noise (k น้อยเกินไป)
และลด underfitting (k มากเกินไป) ได้ดี
""")