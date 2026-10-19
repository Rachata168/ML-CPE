
import sys

print("Python version:", sys.version)
print("Python executable:", sys.executable)


import torch
import ultralytics
import cv2

print("PyTorch version:", torch.__version__)
print("Ultralytics version:", ultralytics.__version__)
print("OpenCV version:", cv2.__version__)

print("CUDA available:", torch.cuda.is_available())
