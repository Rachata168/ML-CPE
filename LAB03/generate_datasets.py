"""
generate_datasets.py
---------------------
สคริปต์นี้ใช้สร้างชุดข้อมูล (dataset) จำลอง "ราคารถยนต์มือสอง" สำหรับ LAB 1
รันไฟล์นี้ครั้งเดียวเพื่อสร้างไฟล์ CSV ที่จะใช้ในแบบฝึกหัด

ใช้ไฟล์เดียว (car_data.csv) ได้ทั้ง Simple และ Multiple Linear Regression:
- Simple  : ใช้แค่คอลัมน์ CarAge -> Price
- Multiple: ใช้ CarAge, Mileage_km, EngineSize_L -> Price
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # ให้ผลลัพธ์เหมือนกันทุกครั้งที่รัน

# ----------------------------------------------------
# Dataset: ราคารถยนต์มือสอง
# ความสัมพันธ์:
#   - อายุรถ (CarAge, ปี)         -> รถยิ่งเก่า ราคายิ่งลด
#   - เลขไมล์ (Mileage_km)         -> วิ่งมาไกล ราคายิ่งลด
#   - ขนาดเครื่องยนต์ (EngineSize_L) -> เครื่องใหญ่ ราคายิ่งสูง
# ----------------------------------------------------
n = 200

car_age = np.random.randint(0, 15, n)                              # อายุรถ 0-15 ปี
mileage = car_age * np.random.uniform(8000, 15000, n) + np.random.normal(0, 5000, n)
mileage = np.clip(mileage, 0, None).round(0)                        # กม.สะสม (สัมพันธ์กับอายุ)
engine_size = np.round(np.random.choice([1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0], n), 1)  # ลิตร

noise = np.random.normal(0, 60000, n)

price = (
    1200000
    - 55000 * car_age
    - 3.5 * mileage
    + 250000 * engine_size
    + noise
)
price = np.clip(price, 100000, None).round(0)   # ราคาต่ำสุดไม่ต่ำกว่า 100,000 บาท

df_car = pd.DataFrame({
    "CarAge": car_age,
    "Mileage_km": mileage,
    "EngineSize_L": engine_size,
    "Price": price
})
df_car = df_car.sort_values("CarAge").reset_index(drop=True)
df_car.to_csv("data/car_data.csv", index=False)
print("สร้างไฟล์ data/car_data.csv สำเร็จ (ใช้ได้ทั้ง Simple และ Multiple Linear Regression)")

print("\nเสร็จสิ้น! ตรวจสอบไฟล์ในโฟลเดอร์ data/ ได้เลยครับ")
