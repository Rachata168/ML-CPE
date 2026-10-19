
from ultralytics import YOLO
import csv
from pathlib import Path

# โฟลเดอร์ที่เก็บโมเดลจริง
base = Path("runs/detect/runs/comparison")
epoch_list = [5, 10, 20]

results_folder = Path("comparison_results")
results_folder.mkdir(exist_ok=True)
csv_file = results_folder / "epoch_comparison.csv"

with open(csv_file, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Epochs", "Precision", "Recall", "mAP50", "mAP50-95"
    ])

    for epochs in epoch_list:
        model_path = (
            base / f"yolo_{epochs}_epochs" / "weights" / "best.pt"
        )

        if not model_path.exists():
            print(f"Skipping {epochs} epochs: model not found")
            continue

        print(f"\nEvaluating {epochs} epochs: {model_path}")
        model = YOLO(str(model_path))

        metrics = model.val(
            data="coco128.yaml",
            imgsz=640,
            batch=8
        )

        writer.writerow([
            epochs,
            metrics.box.mp,
            metrics.box.mr,
            metrics.box.map50,
            metrics.box.map
        ])
        file.flush()

        print(f"Saved results for {epochs} epochs")

print("\nComparison completed!")
print("Results saved to:", csv_file)
