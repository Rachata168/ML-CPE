from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="coco128.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    project="runs/train",
    name="yolo_10_epochs"
)

print("Training completed!")
