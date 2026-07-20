"""
ใบงานที่ 2: Data Preprocessing
วิชา Machine Learning (04-624-201)
Dataset: Titanic (จาก Kaggle - train.csv)

วิธีใช้:
1. เอาไฟล์ train.csv ไปวางในโฟลเดอร์เดียวกับไฟล์ .py นี้
2. รันในเทอร์มินัลด้วยคำสั่ง: python data_preprocessing_lab2.py
3. กราฟจะถูกเซฟเป็นไฟล์ .png ในโฟลเดอร์เดียวกัน (histogram.png, heatmap.png)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LAB 1: Dataset Exploration
# ============================================================

# ---- โหลดข้อมูลจากไฟล์ train.csv (Titanic dataset จาก Kaggle) ----
df = pd.read_csv("train.csv")

# คอลัมน์ในไฟล์นี้คือ:
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
target_col = "Survived"  # คอลัมน์เป้าหมาย (label) ที่จะทำนาย

section("1. Load Dataset")
print(df.head())

section("2. Display Shape")
print("จำนวนแถว (rows), จำนวนคอลัมน์ (columns):", df.shape)

section("3. Display Data Types")
print(df.dtypes)

section("4. Display Summary Statistics")
print(df.describe(include="all"))

section("5. Display Missing Values")
missing = df.isnull().sum()
print(missing[missing > 0])
print("\nเปอร์เซ็นต์ข้อมูลที่หายไป:")
print((missing[missing > 0] / len(df) * 100).round(2).astype(str) + " %")

section("6. Display Duplicate Records")
dup_count = df.duplicated().sum()
print(f"จำนวนแถวที่ซ้ำกัน: {dup_count}")
print(df[df.duplicated()])

section("7. Display Class Distribution")
if target_col in df.columns:
    print(df[target_col].value_counts())
    print((df[target_col].value_counts(normalize=True) * 100).round(2), "%")
else:
    print(f"ไม่พบคอลัมน์ '{target_col}'")


# ============================================================
# LAB 2: Data Visualization
# ============================================================

section("8. Histogram")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.tight_layout()
plt.savefig("histogram.png", dpi=150)
plt.close()
print("บันทึกกราฟ histogram.png แล้ว")

section("9. Correlation Heatmap")
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.close()
print("บันทึกกราฟ heatmap.png แล้ว")


# ============================================================
# Part 3: Data Cleaning
# ============================================================

df_clean = df.copy()

section("10. Missing Value Handling")

# Cabin หายไปเกิน 70% ของข้อมูล -> คอลัมน์นี้ไม่มีประโยชน์พอ ให้ตัดทิ้งไปเลย
# Name, Ticket, PassengerId ก็ไม่ใช่ feature ที่มีประโยชน์กับโมเดล (เป็นแค่ตัวระบุ/ข้อความอิสระ) -> ตัดทิ้งเช่นกัน
cols_to_drop = ["Cabin", "Name", "Ticket", "PassengerId"]
df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
print(f"ตัดคอลัมน์ที่ไม่จำเป็นออก: {cols_to_drop}")

# Age มีค่าหายไปบางส่วน -> เติมด้วยค่ามัธยฐาน (median) เพราะทนต่อค่าผิดปกติมากกว่า mean
if "Age" in df_clean.columns:
    df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

# Embarked หายไปไม่กี่แถว -> เติมด้วยค่าที่พบบ่อยที่สุด (mode)
if "Embarked" in df_clean.columns:
    df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])

# Fare (ถ้ามีหายไป เช่นในไฟล์ test.csv) -> เติมด้วย median เช่นกัน
if "Fare" in df_clean.columns:
    df_clean["Fare"] = df_clean["Fare"].fillna(df_clean["Fare"].median())

print("จำนวน missing values หลังแก้ไข:")
print(df_clean.isnull().sum().sum())

section("11. Duplicate Removal")
before = len(df_clean)
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
after = len(df_clean)
print(f"ลบข้อมูลซ้ำไปทั้งหมด {before - after} แถว (จาก {before} เหลือ {after})")

section("12. Incorrect Data Correction")
# ตัวอย่าง: ตรวจสอบค่าอายุที่ผิดปกติ (ติดลบ หรือมากเกินไป)
if "Age" in df_clean.columns:
    invalid_age = df_clean[(df_clean["Age"] < 0) | (df_clean["Age"] > 100)]
    print(f"พบข้อมูลอายุที่ผิดปกติ: {len(invalid_age)} แถว")
    df_clean.loc[(df_clean["Age"] < 0) | (df_clean["Age"] > 100), "Age"] = df_clean["Age"].median()
    print("แก้ไขค่าผิดปกติด้วยค่ามัธยฐานเรียบร้อยแล้ว")

section("13. Data Type Conversion")
print("ชนิดข้อมูลก่อนแปลง:")
print(df_clean.dtypes)
# แปลงคอลัมน์ categorical (Sex, Embarked, Pclass, Survived) ให้เป็น category type
for col in ["Sex", "Embarked", "Pclass", "Survived"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype("category")
print("\nชนิดข้อมูลหลังแปลง:")
print(df_clean.dtypes)

section("14. Compare Mean vs Median (ก่อน vs หลัง Cleaning)")
for col in ["Age", "Fare", "SibSp", "Parch"]:
    if col in df.columns and col in df_clean.columns:
        mean_before, median_before = df[col].mean(), df[col].median()
        mean_after, median_after = df_clean[col].mean(), df_clean[col].median()
        print(f"\nคอลัมน์: {col}")
        print(f"  ก่อน  -> Mean: {mean_before:.3f}, Median: {median_before:.3f}")
        print(f"  หลัง  -> Mean: {mean_after:.3f}, Median: {median_after:.3f}")


# ============================================================
# Part 4: Feature Engineering
# ============================================================

section("15. Label Encoding")
df_label = df_clean.copy()
categorical_cols = df_label.select_dtypes(include=["object", "category"]).columns
categorical_cols = [c for c in categorical_cols if c != target_col]  # ไม่ encode target ในตัวอย่างนี้

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_label[col] = le.fit_transform(df_label[col].astype(str))
    label_encoders[col] = le

print("ตัวอย่างข้อมูลหลังทำ Label Encoding:")
print(df_label[categorical_cols].head())

section("16. One-Hot Encoding")
df_onehot = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
print("ตัวอย่างข้อมูลหลังทำ One-Hot Encoding:")
print(df_onehot.head())
print(f"\nจำนวนคอลัมน์ก่อน One-Hot: {df_clean.shape[1]}, หลัง One-Hot: {df_onehot.shape[1]}")

section("17. Feature Scaling / Normalization")
scale_cols = [c for c in ["Age", "Fare", "SibSp", "Parch"] if c in df_clean.columns]

scaler_std = StandardScaler()
scaler_mm = MinMaxScaler()

scaled_std = pd.DataFrame(
    scaler_std.fit_transform(df_clean[scale_cols]),
    columns=[f"{c}_standard" for c in scale_cols],
)
scaled_mm = pd.DataFrame(
    scaler_mm.fit_transform(df_clean[scale_cols]),
    columns=[f"{c}_minmax" for c in scale_cols],
)

print("ตัวอย่าง Standard Scaling:")
print(scaled_std.head())
print("\nตัวอย่าง Min-Max Scaling:")
print(scaled_mm.head())

# ============================================================
# บันทึกไฟล์ผลลัพธ์
# ============================================================
df_clean.to_csv("cleaned_data.csv", index=False)
df_onehot.to_csv("data_onehot_encoded.csv", index=False)
print("\nบันทึกไฟล์ cleaned_data.csv และ data_onehot_encoded.csv เรียบร้อยแล้ว")
