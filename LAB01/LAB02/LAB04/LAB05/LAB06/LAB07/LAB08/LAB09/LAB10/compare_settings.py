
from ultralytics import YOLO
import csv
from pathlib import Path

# ตำแหน่งโมเดลที่ฝึกเสร็จแล้ว
base = Path("runs/detect/runs/settings")

experiments = [
    {"name": "imgsz_416", "imgsz": 416, "batch": 8},
    {"name": "imgsz_640", "imgsz": 640, "batch": 8},
    {"name": "batch_16", "imgsz": 640, "batch": 16},
]

output_dir = Path("comparison_results")
output_dir.mkdir(exist_ok=True)
csv_path = output_dir / "settings_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Configuration", "Image Size", "Batch Size",
        "Precision", "Recall", "mAP50", "mAP50-95"
    ])

    for exp in experiments:
        model_path = (
            base / exp["name"] / "weights" / "best.pt"
        )

        if not model_path.exists():
            print("Model not found:", model_path)
            continue

        print(f"\nEvaluating {exp['name']}")
        model = YOLO(str(model_path))

        metrics = model.val(
            data="coco128.yaml",
            imgsz=exp["imgsz"],
            batch=exp["batch"]
        )

        writer.writerow([
            exp["name"],
            exp["imgsz"],
            exp["batch"],
            metrics.box.mp,
            metrics.box.mr,
            metrics.box.map50,
            metrics.box.map
        ])
        f.flush()

        print("Saved:", exp["name"])

print("\nEvaluation completed!")
print("Results saved to:", csv_path)
