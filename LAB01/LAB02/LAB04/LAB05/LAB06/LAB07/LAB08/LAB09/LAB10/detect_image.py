
from pathlib import Path
from ultralytics import YOLO

# ค้นหาโมเดลที่ฝึกเสร็จแล้ว
runs_folder = Path("runs")
model_files = list(runs_folder.rglob("best.pt"))

if not model_files:
    raise FileNotFoundError(
        "ไม่พบ best.pt ในโฟลเดอร์ runs กรุณาตรวจสอบผลการฝึกโมเดล"
    )

# แสดงโมเดลที่พบทั้งหมด
print("Available models:")
for i, path in enumerate(model_files, start=1):
    print(f"{i}. {path}")

# เลือกโมเดลตัวแรกที่พบ
model_path = model_files[0]
print("\nUsing model:", model_path)

model = YOLO(str(model_path))

# ค้นหารูปภาพสำหรับทดสอบ
image_folder = Path("test_images")
image_folder.mkdir(exist_ok=True)

image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
    image_files.extend(image_folder.glob(ext))

print("Found images:", len(image_files))

if image_files:
    model.predict(
        source=[str(path) for path in image_files],
        imgsz=640,
        conf=0.25,
        save=True
    )
    print("Detection completed!")
else:
    print("Please put an image in the test_images folder.")
