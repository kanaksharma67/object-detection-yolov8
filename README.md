# 🚀 YOLOv8 Object Detection Project

This repository contains a simple and clear implementation of training and using the YOLOv8 model (from the [Ultralytics](https://github.com/ultralytics/ultralytics) library) for object detection tasks using custom datasets.

---

### ✅ Prerequisites

- Python 3.10+
- Git
- NVIDIA GPU (recommended) with CUDA installed

---

## 📂 Project Structure

```plaintext
.
├── data.yaml                 # Dataset configuration file
├── test_image.jpg           # Image you want to test
├── detect.py                # Script for inference using trained model
├── train.py                 # Script to start training
├── runs/                    # YOLO's default output folder
│   └── detect/
│       └── train/           # Contains weights and logs after training
│           └── weights/
│               └── best.pt  # Your trained model weights
└── README.md                # This file
