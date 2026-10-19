
from pathlib import Path
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกไว้ 20 Epochs
model_path = Path(
    "runs/detect/runs/comparison/yolo_20_epochs/weights/best.pt"
)

if not model_path.exists():
    raise FileNotFoundError(f"Cannot find model: {model_path}")

model = YOLO(str(model_path))

# โฟลเดอร์เก็บวิดีโอที่ต้องการทดสอบ
video_folder = Path("test_videos")
video_folder.mkdir(exist_ok=True)

video_files = (
    list(video_folder.glob("*.mp4"))
    + list(video_folder.glob("*.avi"))
    + list(video_folder.glob("*.mov"))
)

if not video_files:
    print("Please put a video file in the test_videos folder.")
else:
    for video in video_files:
        print("Detecting:", video.name)

        model.predict(
            source=str(video),
            imgsz=640,
            conf=0.25,
            save=True
        )

    print("Video detection completed!")
    print("Check the runs/detect folder for output videos.")
