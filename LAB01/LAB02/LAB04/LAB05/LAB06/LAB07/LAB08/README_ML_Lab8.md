# ML Lab 8 - Deep Convolutional Neural Network (DCNN)

## ชื่อแลป
Deep Convolutional Neural Network และการประยุกต์ใช้งาน

## วัตถุประสงค์
แลปนี้เป็นการทดลองใช้ Deep Convolutional Neural Network (DCNN) สำหรับจำแนกภาพจากชุดข้อมูล MNIST โดยมีการเตรียมข้อมูล สร้างโมเดล ฝึกโมเดลด้วยจำนวน Epoch ที่แตกต่างกัน เปรียบเทียบโครงสร้างของโมเดล และประเมินผลการทำนาย

## Dataset
ใช้ชุดข้อมูล **MNIST** ซึ่งเป็นภาพตัวเลขเขียนด้วยลายมือ ตั้งแต่ 0 ถึง 9

- Training images: 60,000 ภาพ
- Test images: 10,000 ภาพ
- ขนาดภาพ: 28 x 28 pixels
- จำนวนคลาส: 10 คลาส (0-9)

## การเตรียมข้อมูล
ข้อมูลภาพถูกปรับค่าพิกเซลให้อยู่ในช่วง 0-1 โดยหารด้วย 255 และเพิ่ม Channel Dimension เพื่อให้เหมาะกับการนำไปใช้กับ CNN

รูปแบบข้อมูลหลังเตรียมข้อมูล:
- Training data: `(60000, 28, 28, 1)`
- Testing data: `(10000, 28, 28, 1)`

## โครงสร้างโปรเจกต์

```text
ML_Lab8
│
├── load_dataset.py
├── split_standardize.py
├── build_dcnn.py
├── train_epochs.py
├── compare_models.py
├── evaluate_predict.py
└── README.md
```

## รายละเอียดแต่ละไฟล์

### `load_dataset.py`
ใช้สำหรับโหลดชุดข้อมูล MNIST และตรวจสอบขนาดของ Training Set และ Testing Set

### `split_standardize.py`
ใช้สำหรับเตรียมข้อมูลก่อนการฝึกโมเดล โดย
- แยก Training และ Testing data
- Standardize ค่าพิกเซลให้อยู่ในช่วง 0-1
- ปรับรูปแบบข้อมูลให้มี Channel สำหรับ CNN

### `build_dcnn.py`
ใช้สำหรับสร้าง DCNN Model และแสดงรายละเอียดของโมเดล

โครงสร้างหลักประกอบด้วย
- Convolutional Layer
- Max Pooling Layer
- Convolutional Layer
- Max Pooling Layer
- Flatten Layer
- Dense Layer
- Output Layer จำนวน 10 neurons

### `train_epochs.py`
ใช้ทดลองฝึกโมเดลด้วยจำนวน Epoch ที่แตกต่างกัน ได้แก่

- 5 Epochs
- 10 Epochs
- 15 Epochs

และแสดงผล
- Training Accuracy
- Training Loss
- Validation Accuracy
- Validation Loss
- Test Accuracy
- Test Loss

### `compare_models.py`
ใช้เปรียบเทียบ DCNN ที่มีโครงสร้างต่างกัน

**Model 1**
- 1 Convolutional Layer
- Dense Layer 32 neurons

**Model 2**
- 2 Convolutional Layers
- Dense Layer 64 neurons

ทั้งสองโมเดลฝึกด้วย 5 Epochs และนำ Test Accuracy มาเปรียบเทียบกัน

### `evaluate_predict.py`
ใช้สำหรับ
- Train โมเดล
- ประเมินผลบน Test Dataset
- แสดง Test Accuracy และ Test Loss
- แสดงกราฟ Training/Validation Accuracy
- แสดงกราฟ Training/Validation Loss
- ทำนายภาพตัวอย่าง 10 ภาพจาก Test Dataset

## Libraries ที่ใช้

- Python
- TensorFlow / Keras
- Matplotlib

ติดตั้งด้วยคำสั่ง:

```bash
pip install tensorflow matplotlib
```

## วิธีรันโปรแกรม

เปิด Terminal ใน VS Code แล้วเข้าไปในโฟลเดอร์ `ML_Lab8`

จากนั้นสามารถรันแต่ละส่วนแยกกันได้ดังนี้:

```bash
python load_dataset.py
```

```bash
python split_standardize.py
```

```bash
python build_dcnn.py
```

```bash
python train_epochs.py
```

```bash
python compare_models.py
```

```bash
python evaluate_predict.py
```

## Output ที่ได้จากการทดลอง

ผลลัพธ์จากแลปประกอบด้วย

- ขนาดของ Dataset หลังโหลดและเตรียมข้อมูล
- โครงสร้างของ DCNN Model
- Accuracy ของแต่ละจำนวน Epoch
- Training และ Validation Accuracy
- Training และ Validation Loss
- Test Accuracy และ Test Loss
- Accuracy ของแต่ละ DCNN Configuration
- ผลการ Prediction ของภาพจาก Test Dataset

## สรุป
การทดลองนี้แสดงให้เห็นการประยุกต์ใช้ DCNN ในการจำแนกภาพตัวเลขจาก MNIST ตั้งแต่ขั้นตอนการโหลดและเตรียมข้อมูล การสร้างโครงสร้าง CNN การฝึกโมเดลด้วยจำนวน Epoch ที่แตกต่างกัน ไปจนถึงการเปรียบเทียบโมเดลและประเมินประสิทธิภาพด้วย Accuracy และ Loss
