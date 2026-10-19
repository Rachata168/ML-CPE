
from pathlib import Path
from ultralytics import YOLO


dataset_path = Path("coco128")

if not dataset_path.exists():
    model = YOLO("yolo11n.pt")
    model.train(
        data="coco128.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
        project="runs/dataset_test",
        name="check"
    )

print("Dataset preparation step completed.")
