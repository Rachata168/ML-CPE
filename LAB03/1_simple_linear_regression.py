"""
1_simple_linear_regression.py
------------------------------
LAB 1.1: Simple Linear Regression
โจทย์: ทำนาย "ราคารถยนต์มือสอง (Price)" จาก "อายุรถ (CarAge)"
      -> มีตัวแปรต้น (X) เพียงตัวเดียว จึงเรียกว่า "Simple" Linear Regression
"""

# ----------------------------------------------------
# STEP 1: Import ไลบรารีที่ต้องใช้
# ----------------------------------------------------
import pandas as pd                                  # จัดการตาราง/ข้อมูล
import matplotlib.pyplot as plt                       # วาดกราฟ
from sklearn.model_selection import train_test_split  # แบ่งข้อมูล train/test
from sklearn.linear_model import LinearRegression     # โมเดล Regression
from sklearn.metrics import mean_squared_error, r2_score  # วัดประสิทธิภาพ
import numpy as np

# ----------------------------------------------------
# STEP 2: โหลดข้อมูล (Load Dataset)
# ----------------------------------------------------
df = pd.read_csv("data/car_data.csv")

print("ตัวอย่างข้อมูล 5 แถวแรก:")
print(df.head())
print("\nจำนวนข้อมูลทั้งหมด:", df.shape)

# ----------------------------------------------------
# STEP 3: สำรวจข้อมูลด้วยกราฟ (EDA - Exploratory Data Analysis)
# ----------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(df["CarAge"], df["Price"], color="steelblue")
plt.title("Relationship between Car Age and Price")
plt.xlabel("Car Age (years)")
plt.ylabel("Price (THB)")
plt.grid(alpha=0.3)
plt.savefig("output_1_scatter.png", dpi=120)
plt.show()

# ----------------------------------------------------
# STEP 4: แยก Feature (X) และ Target (y)
# ----------------------------------------------------
X = df[["CarAge"]]   # ต้องเป็น 2 มิติ (DataFrame) เวลาใส่ให้ sklearn
y = df["Price"]       # เป็น 1 มิติ (Series) ก็พอ

# ----------------------------------------------------
# STEP 5: แบ่งข้อมูลเป็น Training set และ Testing set
#         (สอนโมเดลด้วย train, แล้วเอา test มาวัดผลจริง)
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42   # 80% train, 20% test
)
print(f"\nข้อมูล Train: {len(X_train)} แถว | ข้อมูล Test: {len(X_test)} แถว")

# ----------------------------------------------------
# STEP 6: สร้างและฝึกโมเดล (Train Model)
# ----------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nสมการที่โมเดลเรียนรู้ได้: Price = {model.intercept_:.2f} + ({model.coef_[0]:.2f}) * CarAge")

# ----------------------------------------------------
# STEP 7: ทำนายผลด้วยข้อมูล Test
# ----------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------
# STEP 8: ประเมินผลโมเดล (Evaluation)
# ----------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== ผลการประเมินโมเดล =====")
print(f"MSE  (Mean Squared Error)      : {mse:,.2f}")
print(f"RMSE (Root Mean Squared Error) : {rmse:,.2f}")
print(f"R^2 Score (ยิ่งใกล้ 1 ยิ่งดี)   : {r2:.4f}")

# ----------------------------------------------------
# STEP 9: วาดกราฟเปรียบเทียบเส้น Regression กับข้อมูลจริง
# ----------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(X_train, y_train, color="steelblue", label="Training data")
plt.scatter(X_test, y_test, color="orange", label="Testing data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression line")
plt.title("Simple Linear Regression: Car Price Prediction")
plt.xlabel("Car Age (years)")
plt.ylabel("Price (THB)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("output_1_regression_line.png", dpi=120)
plt.show()

# ----------------------------------------------------
# STEP 10: ลองทำนายค่าใหม่ (ตัวอย่างการใช้งานจริง)
# ----------------------------------------------------
new_car_age = [[5]]  # สมมติรถอายุ 5 ปี
predicted_price = model.predict(new_car_age)
print(f"\nถ้ารถอายุ 5 ปี โมเดลทำนายราคา = {predicted_price[0]:,.2f} บาท")
