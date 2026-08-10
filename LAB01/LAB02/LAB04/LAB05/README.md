# ใบงานที่ 5 — Support Vector Machine (SVM)

## LEB: SVM on a Dataset of Your Choice

### Objective
- Apply Support Vector Machine (SVM) to classify a dataset of your choice
- Compare the performance of different SVM kernels

### Dataset
ใช้ **Iris Dataset** จาก `sklearn.datasets` ประกอบด้วยข้อมูลดอกไอริส 3 สายพันธุ์ (setosa, versicolor, virginica) จำนวน 150 ตัวอย่าง 4 features ได้แก่ sepal length, sepal width, petal length, petal width

### วิธีการทำงาน (Contents)
1. **โหลดข้อมูล** — โหลด Iris dataset จาก scikit-learn
2. **สำรวจข้อมูล (Explore)** — ตรวจสอบข้อมูลด้วย `.head()`, `.describe()`, จำนวนข้อมูลแต่ละ class และแสดง pairplot เพื่อดูความสัมพันธ์ระหว่าง features
3. **แบ่งข้อมูล** — แบ่งเป็น training set (80%) และ testing set (20%) แบบ stratified
4. **Standardize** — ปรับ scale ข้อมูลด้วย `StandardScaler` (fit บน training set แล้ว transform ทั้ง train และ test)
5. **เทรนโมเดล SVM** — เทรนด้วย 3 kernel ได้แก่ Linear, Polynomial (poly), RBF
6. **ประเมินผล** — วัดผลด้วย accuracy, classification report (precision, recall, f1-score) และ confusion matrix ของแต่ละ kernel
7. **เปรียบเทียบผล** — สรุปและเปรียบเทียบ accuracy ของทั้ง 3 kernel ด้วยกราฟแท่ง
8. **ทำนายผล** — เลือก kernel ที่ให้ accuracy สูงที่สุดมาทำนายตัวอย่างข้อมูลจริง

### วิธีรัน
```bash
python -m pip install scikit-learn pandas matplotlib seaborn numpy
python svm_lab.py
```

### ไฟล์ในโปรเจกต์
| ไฟล์ | คำอธิบาย |
|------|----------|
| `svm_lab.py` | โค้ดหลักของ LAB |
| `pairplot.png` | กราฟสำรวจความสัมพันธ์ระหว่าง features |
| `confusion_matrix_linear.png` | Confusion matrix ของ Linear kernel |
| `confusion_matrix_poly.png` | Confusion matrix ของ Polynomial kernel |
| `confusion_matrix_rbf.png` | Confusion matrix ของ RBF kernel |
| `kernel_comparison.png` | กราฟเปรียบเทียบ accuracy ของ 3 kernel |

## ผลลัพธ์ (Output)

### Accuracy scores
| Kernel | Accuracy |
|--------|----------|
| Linear | 1.0000 (100%) |
| Polynomial | 0.9000 (90%) |
| RBF | 0.9667 (96.67%) |

### สรุปผล
- **Linear kernel** ให้ผลลัพธ์ดีที่สุด แยกทั้ง 3 class ได้ถูกต้อง 100% เนื่องจาก Iris dataset มีลักษณะที่ค่อนข้าง linearly separable โดยเฉพาะ class setosa ที่แยกออกจากอีก 2 class ได้ชัดเจนมาก
- **RBF kernel** ให้ผลรองลงมา มีความผิดพลาดเล็กน้อยในการแยก versicolor กับ virginica
- **Polynomial kernel** ให้ผลต่ำที่สุด สับสนระหว่าง versicolor กับ virginica มากที่สุด (ทายผิด 3 จาก 10 ตัวอย่างของ virginica)

### Predictions ตัวอย่าง (ใช้ Linear kernel)
ทำนายตัวอย่างข้อมูล 5 ตัวจาก test set ได้ถูกต้องทั้งหมด (5/5)
