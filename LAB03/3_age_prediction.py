"""
3_age_prediction.py
---------------------
LAB 1.3: Age Prediction จากภาพใบหน้า (ใช้ PCA ลดจำนวนคุณลักษณะ)

**สำคัญ: ต้องดาวน์โหลด Dataset ก่อนรันไฟล์นี้**

วิธีดาวน์โหลด Dataset (UTKFace):
1. เข้าเว็บ Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new
   (ถ้ายังไม่มีบัญชี ให้สมัครฟรีก่อน)
2. กดปุ่ม Download จะได้ไฟล์ archive.zip
3. แตกไฟล์ zip แล้วจะได้โฟลเดอร์ชื่อ utkface_aligned_cropped/UTKFace
   ซึ่งข้างในมีรูปภาพใบหน้าหลายพันรูป
4. คัดลอกโฟลเดอร์ UTKFace มาไว้ในโปรเจกต์นี้ ให้ path เป็น:
   lab1/data/UTKFace/*.jpg
   (ชื่อไฟล์แต่ละรูปจะขึ้นต้นด้วยอายุ เช่น "25_0_0_20170116174525125.jpg"
    ตัวเลขแรก "25" คืออายุของคนในภาพ)

ถ้ายังไม่สะดวกดาวน์โหลด dataset จริง ให้รัน generate_fake_faces.py
ก่อน เพื่อสร้างภาพจำลองไว้ทดสอบโค้ดนี้ได้เลย (ดูคำอธิบายท้ายไฟล์)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------------------
# STEP 1: กำหนด path ของโฟลเดอร์ dataset
# ----------------------------------------------------
DATA_DIR = "data/UTKFace"
IMG_SIZE = 64   # ย่อภาพให้เล็กลงเหลือ 64x64 พิกเซล เพื่อลดเวลาประมวลผล

# ----------------------------------------------------
# STEP 2: ฟังก์ชันโหลดภาพทั้งหมดพร้อมดึง "อายุ" จากชื่อไฟล์
# ----------------------------------------------------
def load_utkface_data(data_dir, img_size=64, max_images=3000):
    images = []
    ages = []

    files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
    files = files[:max_images]   # จำกัดจำนวนภาพ ไม่ให้โหลดนานเกินไป

    for filename in files:
        try:
            # ชื่อไฟล์รูปแบบ: age_gender_race_date.jpg
            age = int(re.match(r"^(\d+)_", filename).group(1))

            img_path = os.path.join(data_dir, filename)
            img = Image.open(img_path).convert("L")          # แปลงเป็นภาพขาวดำ (Grayscale)
            img = img.resize((img_size, img_size))            # ย่อขนาดภาพ
            img_array = np.array(img).flatten()               # แปลงภาพ 2D -> เวกเตอร์ 1D

            images.append(img_array)
            ages.append(age)
        except Exception as e:
            continue   # ข้ามไฟล์ที่เสียหรืออ่านไม่ได้

    X = np.array(images)
    y = np.array(ages)
    return X, y

print("กำลังโหลดข้อมูลภาพใบหน้า...")
X_raw, y_age = load_utkface_data(DATA_DIR, img_size=IMG_SIZE, max_images=3000)
print(f"โหลดภาพสำเร็จ: {X_raw.shape[0]} ภาพ | แต่ละภาพมี {X_raw.shape[1]} พิกเซล (ก่อนทำ PCA)")

# ----------------------------------------------------
# STEP 3: แบ่งข้อมูล Train / Test ก่อนทำ PCA
#         (ต้อง fit PCA ด้วย train เท่านั้น ป้องกัน Data Leakage)
# ----------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_age, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# STEP 4: ปรับสเกลข้อมูล (Standardization)
#         PCA ไวต่อสเกลของข้อมูล จึงควร scale ก่อนเสมอ
# ----------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ----------------------------------------------------
# STEP 5: ทำ PCA (Principal Component Analysis)
#         ลดจำนวนคุณลักษณะจากหลายพันพิกเซล เหลือแค่ n_components ตัว
#         โดยยังคงความแปรปรวน (variance) ของข้อมูลไว้ให้มากที่สุด
# ----------------------------------------------------
n_components = 100   # ลองปรับตัวเลขนี้ดูได้ (เช่น 50, 100, 150)
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"\nใช้ {n_components} Principal Components")
print(f"อธิบายความแปรปรวนของข้อมูลได้ {explained_var:.2f}%")

# ----------------------------------------------------
# STEP 6: ฝึกโมเดล Linear Regression บนข้อมูลที่ทำ PCA แล้ว
# ----------------------------------------------------
model = LinearRegression()
model.fit(X_train_pca, y_train)

# ----------------------------------------------------
# STEP 7: ทำนายและประเมินผล
# ----------------------------------------------------
y_pred = model.predict(X_test_pca)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== ผลการประเมินโมเดล Age Prediction =====")
print(f"MAE (คลาดเคลื่อนเฉลี่ย) : {mae:.2f} ปี")
print(f"R^2 Score               : {r2:.4f}")

# ----------------------------------------------------
# STEP 8: กราฟเปรียบเทียบอายุจริง vs อายุที่ทำนาย
# ----------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
plt.plot([y_age.min(), y_age.max()], [y_age.min(), y_age.max()], color="red", linewidth=2)
plt.title("Age Prediction: Actual vs Predicted Age")
plt.xlabel("Actual Age (years)")
plt.ylabel("Predicted Age (years)")
plt.grid(alpha=0.3)
plt.savefig("output_3_age_prediction.png", dpi=120)
plt.show()

# ----------------------------------------------------
# STEP 9: กราฟ Explained Variance ของ PCA (โบนัส เข้าใจ PCA มากขึ้น)
# ----------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("PCA Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.grid(alpha=0.3)
plt.savefig("output_3_pca_variance.png", dpi=120)
plt.show()

print("\nเสร็จสิ้น! ดูภาพผลลัพธ์ได้ที่ output_3_age_prediction.png และ output_3_pca_variance.png")
