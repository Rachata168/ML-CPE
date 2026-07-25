"""
2_multiple_linear_regression.py
---------------------------------
LAB 1.2: Multiple Linear Regression
โจทย์: ทำนาย "ราคารถยนต์มือสอง (Price)" จากหลายตัวแปร
      -> อายุรถ (CarAge), เลขไมล์สะสม (Mileage_km), ขนาดเครื่องยนต์ (EngineSize_L)
      -> มีตัวแปรต้นมากกว่า 1 ตัว จึงเรียกว่า "Multiple" Linear Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------
# STEP 1: โหลดข้อมูล
# ----------------------------------------------------
df = pd.read_csv("data/car_data.csv")
print("ตัวอย่างข้อมูล 5 แถวแรก:")
print(df.head())
print("\nสถิติเบื้องต้น:")
print(df.describe())

# ----------------------------------------------------
# STEP 2: ดูความสัมพันธ์ระหว่างตัวแปร (Correlation)
#         ช่วยให้เข้าใจว่าตัวแปรไหนมีผลต่อราคาบ้านมากน้อยแค่ไหน
# ----------------------------------------------------
print("\nค่า Correlation กับ Price:")
print(df.corr()["Price"].sort_values(ascending=False))

# ----------------------------------------------------
# STEP 3: แยก Feature (X) และ Target (y)
#         ครั้งนี้ X มีหลายคอลัมน์ (Multiple Features)
# ----------------------------------------------------
X = df[["CarAge", "Mileage_km", "EngineSize_L"]]
y = df["Price"]

# ----------------------------------------------------
# STEP 4: แบ่ง Train / Test
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# STEP 5: สร้างและฝึกโมเดล
# ----------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\n===== สมการที่โมเดลเรียนรู้ได้ =====")
print(f"Price = {model.intercept_:,.2f}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"        + ({coef:,.2f}) * {feature}")

# ----------------------------------------------------
# STEP 6: ทำนายและประเมินผล
# ----------------------------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== ผลการประเมินโมเดล =====")
print(f"RMSE      : {rmse:,.2f} บาท")
print(f"R^2 Score : {r2:.4f}")

# ----------------------------------------------------
# STEP 7: วาดกราฟเปรียบเทียบ "ค่าจริง" กับ "ค่าที่ทำนาย"
#         (Multiple Regression มีหลายมิติ วาดเส้นตรงไม่ได้ตรงๆ
#          จึงนิยมวาดกราฟ Actual vs Predicted แทน)
# ----------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color="steelblue", alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2, label="Perfect prediction line")
plt.title("Multiple Linear Regression: Actual vs Predicted Car Price")
plt.xlabel("Actual Price (THB)")
plt.ylabel("Predicted Price (THB)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("output_2_actual_vs_predicted.png", dpi=120)
plt.show()

# ----------------------------------------------------
# STEP 8: ทดลองทำนายรถยนต์คันใหม่
# ----------------------------------------------------
new_car = pd.DataFrame({
    "CarAge": [5],
    "Mileage_km": [60000],
    "EngineSize_L": [1.8]
})
predicted_price = model.predict(new_car)
print(f"\nรถอายุ 5 ปี วิ่งมาแล้ว 60,000 กม. เครื่องยนต์ 1.8 ลิตร -> ราคาที่ทำนาย = {predicted_price[0]:,.2f} บาท")

# ----------------------------------------------------
# STEP 9 (โบนัส): เปรียบเทียบ Simple vs Multiple Regression
#          ใช้แค่ CarAge อย่างเดียวเทียบกับใช้ 3 ตัวแปร
# ----------------------------------------------------
simple_model = LinearRegression()
simple_model.fit(X_train[["CarAge"]], y_train)
y_pred_simple = simple_model.predict(X_test[["CarAge"]])
r2_simple = r2_score(y_test, y_pred_simple)

print("\n===== เปรียบเทียบโมเดล (สำหรับ LAB 3) =====")
print(f"Simple Linear Regression (CarAge เท่านั้น)  R^2 = {r2_simple:.4f}")
print(f"Multiple Linear Regression (3 ตัวแปร)        R^2 = {r2:.4f}")
