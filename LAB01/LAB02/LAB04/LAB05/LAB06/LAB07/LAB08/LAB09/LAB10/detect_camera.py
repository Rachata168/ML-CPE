
from pathlib import Path
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกไว้ 20 Epochs
model_path = Path(
    "runs/detect/runs/comparison/yolo_20_epochs/weights/best.pt"
)

if not model_path.exists():
    raise FileNotFoundError(f"Cannot find model: {model_path}")

model = YOLO(str(model_path))

print("Starting webcam detection...")
print("Press Q in the camera window to stop.")

# เปิดกล้องเว็บแคม
model.predict(
    source=0,
    imgsz=640,
    conf=0.25,
    show=True
)

print("Camera detection stopped.")
