# Lab 7: Convolutional Neural Network (CNN) on Fashion-MNIST

## วัตถุประสงค์
1. เข้าใจหลักการทำงานของ Convolutional Neural Network (CNN) และแนวทางการประยุกต์ใช้ในงานจำแนกและพยากรณ์
2. พัฒนาผังการทำงาน ประมวลผล และแสดงผลลัพธ์ เพื่อพัฒนาระบบต้นแบบร่วมกับ CNN ในการจำแนกและพยากรณ์ได้อย่างถูกต้อง

## Dataset
ใช้ **Fashion-MNIST** (ผ่าน `tensorflow.keras.datasets`) ซึ่งเป็นภาพขาวดำขนาด 28x28 พิกเซล จำนวน 10 คลาส (เสื้อผ้า/รองเท้า/กระเป๋า) แบ่งเป็น 60,000 ภาพสำหรับ train และ 10,000 ภาพสำหรับ test

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `load_data.py` | โหลด Fashion-MNIST dataset (train/test) |
| `preprocess.py` | Standardize พิกเซล (0-255 → 0-1), reshape เป็น (28,28,1), one-hot encode labels |
| `build_model.py` | สร้างโครงสร้าง CNN (Conv2D + MaxPooling2D + Dense) แบบปรับ config ได้ (จำนวน conv layers, จำนวน filters) |
| `train.py` | ฟังก์ชันเทรนโมเดล 1 config |
| `compare_configs.py` | รันเทรนหลาย config พร้อมกัน (ต่างจำนวน epochs/layers/filters) แล้วเปรียบเทียบ accuracy |
| `plot_results.py` | วาดกราฟ train vs validation accuracy/loss ต่อ 1 config |
| `plot_compare.py` | วาดกราฟสรุปเปรียบเทียบ accuracy/loss ของทุก config ในภาพเดียว |
| `predict.py` | ทำนายผลตัวอย่างจาก test set พร้อมเทียบกับ label จริง |

## CNN Configurations ที่เปรียบเทียบ

| Epochs | Conv Layers | Filters | Test Accuracy |
|---|---|---|---|
| 5 | 2 | 32 | 0.9035 |
| 10 | 2 | 32 | **0.9069** (ดีที่สุด) |
| 5 | 3 | 32 | 0.8788 |
| 10 | 3 | 64 | 0.8979 |

## วิธีการรัน

1. สร้างและเปิดใช้งาน virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
2. ติดตั้งไลบรารีที่จำเป็น
   ```bash
   python -m pip install tensorflow numpy matplotlib scikit-learn
   ```
3. รันไฟล์หลักไฟล์เดียว (จะเรียกไฟล์อื่นทั้งหมดให้อัตโนมัติ)
   ```bash
   python compare_configs.py
   ```
4. ดูตัวอย่างการทำนายผล
   ```bash
   python predict.py
   ```

## ผลลัพธ์ที่ได้ (Output)
- Accuracy scores ของแต่ละ CNN configuration (ตารางด้านบน)
- กราฟเปรียบเทียบ accuracy ระหว่างจำนวน epochs ต่างกัน (`Validation_Accuracy_per_Config.png`, `Validation_Loss_per_Config.png`)
- กราฟ training/validation accuracy และ loss ต่อ config (`e{epochs}_l{layers}_f{filters}.png` — 4 ไฟล์)
- ตัวอย่างการทำนายผลบน test set พร้อมเทียบกับ label จริง (จากการรัน `predict.py`)

## สรุปผลการวิเคราะห์
Config ที่มี conv layer น้อยกว่า (2 layers) แต่เทรนนานขึ้น (10 epochs) ให้ผล test accuracy ดีที่สุด (0.9069) ในขณะที่ config ที่เพิ่มจำนวน conv layer เป็น 3 ชั้น กลับได้ accuracy ต่ำกว่า อาจเนื่องมาจากโมเดลมีความซับซ้อนเกินความจำเป็นสำหรับ dataset ขนาดและความซับซ้อนระดับนี้ หรือต้องการจำนวน epoch ที่มากขึ้นกว่านี้เพื่อให้โมเดลลู่เข้า (converge) ได้ดีกว่านี้ กราฟ train vs validation ของทุก config ไม่แสดงสัญญาณ overfitting ที่ชัดเจน (เส้น train และ val ไม่ห่างกันมาก)
