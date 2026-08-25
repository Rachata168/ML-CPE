# ใบงานที่ 6: Neural Network และการประยุกต์ใช้งาน

## วัตถุประสงค์
1. เข้าใจหลักการทำงานของ Neural Network (NN) และแนวทางการประยุกต์ใช้งานจริงทั้งด้านการจำแนกและการพยากรณ์
2. พัฒนาโปรแกรมประมวลผลและแสดงผลลัพธ์ เพื่อพัฒนาระบบต้นแบบร่วมกับ NN ในการจำแนก/พยากรณ์ได้อย่างถูกต้อง

## Dataset ที่ใช้
**Iris Dataset** จาก `sklearn.datasets`
- จำนวนตัวอย่าง: 150 samples
- Features: 4 (ความยาว/ความกว้างของกลีบดอกและกลีบเลี้ยง)
- Classes: 3 ชนิด (setosa, versicolor, virginica)

## วิธีการทำงาน (Methodology)
1. **โหลดข้อมูล** จาก Iris dataset
2. **แบ่งข้อมูล** เป็น Training set (80%) และ Testing set (20%) แบบ stratified
3. **Standardize** ฟีเจอร์ด้วย `StandardScaler` ก่อนนำเข้าโมเดล
4. **สร้างโมเดล Neural Network** ด้วย `MLPClassifier` (scikit-learn) จำนวน 4 configuration:

   | Configuration | Hidden Layers | Epoch (max_iter) |
   |---|---|---|
   | 1 | (10,) | 100 |
   | 2 | (10,) | 300 |
   | 3 | (20, 10) | 300 |
   | 4 | (20, 10) | 500 |

5. **Train** แต่ละโมเดลแล้ววัด Train/Test Accuracy
6. **เปรียบเทียบผล** ระหว่าง configuration และจำนวน epoch ที่ต่างกัน
7. **Plot กราฟ** Loss Curve และ Validation Accuracy ของโมเดลที่ดีที่สุด
8. **ทำนายผล (Predict)** บนชุดข้อมูล test และแสดง Classification Report

## ผลลัพธ์ (Results)

| Configuration | Train Acc | Test Acc |
|---|---|---|
| 1 hidden layer (10), epoch=100 | 0.8667 | 0.8333 |
| 1 hidden layer (10), epoch=300 | 0.8667 | 0.8333 |
| 2 hidden layers (20,10), epoch=300 | 0.9500 | 0.9000 |
| 2 hidden layers (20,10), epoch=500 | 0.9500 | 0.9000 |

**โมเดลที่ดีที่สุด:** 2 hidden layers (20, 10), epoch=300 → Test Accuracy = **90%**

**สรุปผล:**
- โมเดลที่มี 2 hidden layers ให้ผลลัพธ์ดีกว่าโมเดลที่มี 1 hidden layer เพราะมี capacity ในการเรียนรู้ pattern ที่ซับซ้อนได้มากกว่า
- การเพิ่มจำนวน epoch จาก 300 เป็น 500 ไม่ได้ทำให้ accuracy เพิ่มขึ้น แสดงว่าโมเดล converge (บรรจบ) แล้วตั้งแต่ epoch ที่น้อยกว่า เพิ่ม epoch เกินจุดนี้ไม่ช่วยและอาจเสี่ยงต่อ overfitting
- Loss curve แสดงว่า training loss ลดลงอย่างต่อเนื่องและ validation accuracy เพิ่มขึ้นแล้วคงที่ ซึ่งเป็นสัญญาณว่าโมเดล train ได้เหมาะสม ไม่ underfit

## ไฟล์ในโปรเจกต์
```
├── neural_network.py     # โค้ดหลักของ Lab
├── nn_loss_curve.png      # กราฟ Loss Curve ของโมเดลที่ดีที่สุด
└── README.md              # ไฟล์นี้
```

## วิธีการรัน
1. ติดตั้ง library ที่จำเป็น:
   ```bash
   python -m pip install scikit-learn matplotlib
   ```
2. รันไฟล์:
   ```bash
   python neural_network.py
   ```
3. ผลลัพธ์จะแสดงในหน้าจอ (comparison table, predictions, classification report) และกราฟ Loss Curve จะถูกเซฟเป็น `nn_loss_curve.png`

## เครื่องมือที่ใช้
- Python 3.x
- scikit-learn (MLPClassifier, StandardScaler, train_test_split)
- matplotlib (สำหรับ plot กราฟ)
