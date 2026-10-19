
from pathlib import Path
from ultralytics import YOLO

# โมเดลที่ฝึกด้วย 20 Epochs
model_path = Path(
    "runs/detect/runs/comparison/yolo_20_epochs/weights/best.pt"
)

if not model_path.exists():
    raise FileNotFoundError(
        f"Cannot find model: {model_path}"
    )

# โหลดโมเดล
print("Loading model:", model_path)
model = YOLO(str(model_path))

# ประเมินประสิทธิภาพ
print("Evaluating model...")
metrics = model.val(
    data="coco128.yaml",
    imgsz=640,
    batch=8
)

# แสดงผล
print("\n===== Model Evaluation Results =====")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print("====================================")

