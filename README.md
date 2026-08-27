# Vehicle Traffic Vision System

A computer vision project completed during a **SDAIA Computer Vision program**.

## Program Name

**Computer Vision for Developers with Ultralytics**

**Session dates:** August 23–27, 2026

**Tutor:** Naif Mersal

**SDAIA Academy GitHub:** https://github.com/SDAIAAcademy

---

## Project Overview

This project applies Ultralytics YOLO to a vehicle and traffic-related computer vision workflow.

The project includes:

- Object detection on an image
- Instance segmentation on an image
- Object tracking in a traffic video
- Model evaluation using the COCO8 validation dataset
- Custom vehicle dataset preparation
- Custom YOLO model fine-tuning

The main implementation and captured execution results are contained in:

`lec1.ipynb`

---

## Pipeline

```text
Input Image
    |
    +--> YOLO Object Detection
    |
    +--> YOLO Instance Segmentation

Traffic Video
    |
    +--> OpenCV Video Capture
    |
    +--> YOLO Object Tracking

COCO8 Dataset
    |
    +--> YOLO Model Validation
    |
    +--> Precision / Recall / mAP

Custom Vehicle Dataset
    |
    +--> Dataset Preparation
    |
    +--> YOLO Fine-Tuning
```

---

# Part 1 — Object Detection and Instance Segmentation

## Object Detection

A pretrained YOLO model is used to perform object detection on:

`bus.jpg`

The model returns detected objects, confidence scores, and bounding-box coordinates in XYXY format.

### Model

`yolo26n.pt`

### Main API

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("bus.jpg")
```

The notebook displays the detection result and prints detection information, including the number of detected objects, class names, confidence scores, and the coordinates of the first detection.

## Instance Segmentation

Instance segmentation is demonstrated using task-specific segmentation weights.

### Model

`yolo26n-seg.pt`

### Main API

```python
segmentation_model = YOLO("yolo26n-seg.pt")
segmentation_results = segmentation_model("bus.jpg")
```

The notebook reports the number and shape of segmentation masks and displays the segmentation result.

---

# Part 2 — Object Tracking

The YOLO model is applied to a traffic video using an OpenCV processing pipeline.

### Input

`road_trafifc.mp4`

The video is read frame by frame with OpenCV. YOLO tracking is then applied to the frames, allowing objects to be followed across the video with tracking IDs.

### Main API

```python
results = model.track(frame, persist=True)
```

The annotated frames are displayed using OpenCV.

The tracking implementation uses:

- YOLO
- OpenCV
- Persistent object tracking
- Bounding boxes
- Object labels
- Tracking IDs

---

# Part 3 — Model Evaluation

The pretrained YOLO model is evaluated using the **COCO8 validation dataset**.

### Main API

```python
results = model.val(data="coco8.yaml")
```

The validation run reports:

- Precision
- Recall
- mAP50
- mAP50-95
- Per-class metrics

## Evaluation Metrics

### Precision

Precision describes how many of the model's predicted detections are correct.

### Recall

Recall describes how many of the objects present in the validation data are successfully detected.

### mAP50

mAP50 is mean Average Precision calculated at an IoU threshold of 0.50.

### mAP50-95

mAP50-95 is mean Average Precision averaged across IoU thresholds from 0.50 through 0.95. It provides a stricter measure of localization quality than mAP50.

### IoU

Intersection over Union (IoU) measures the overlap between a predicted bounding box and its ground-truth bounding box.

### Confidence

The confidence score represents the model's confidence in a detection. Changing the confidence threshold changes the trade-off between false positives and false negatives.

## Failure Analysis

The evaluation code identifies classes with:

- The lowest recall, indicating where the model misses more objects and produces more false negatives.
- The lowest mAP50-95, indicating weaker localization performance at stricter IoU thresholds.

The notebook contains the actual validation metrics and the resulting per-class analysis.

---

# Part 4 — Custom Vehicle Dataset and Training

A real custom vehicle dataset obtained from Kaggle is prepared and used for custom YOLO training.

## Part 4.1 — Dataset Preparation

The source dataset is stored in:

`training_image`

Images are included only when a corresponding YOLO-format label file is available.

The dataset is shuffled using a fixed random seed:

```python
random.seed(42)
```

It is then split into 80% training data and 20% validation data.

### Dataset Size

- Total labeled images: **390**
- Training images: **312**
- Validation images: **78**

### Classes

- Ambulance
- Bus
- Car
- Truck

The prepared dataset follows the YOLO directory structure:

```text
vehicle_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

The dataset configuration is written to:

`vehicle.yaml`

## Part 4.2 — Custom Model Training

A pretrained YOLO model is fine-tuned using the custom vehicle dataset.

### Training Configuration

| Setting | Value |
|---|---|
| Model | `yolo26n.pt` |
| Dataset | `vehicle.yaml` |
| Epochs | 10 |
| Image size | 640 |
| Batch size | 8 |
| Classes | 4 |
| Pretrained | Yes |

### Main API

```python
results = model.train(
    data="vehicle.yaml",
    epochs=10,
    imgsz=640,
    batch=8
)
```

The training run completed successfully, and the training output was captured in the notebook. Ultralytics saved the generated training results.

---

# Dataset Source

The custom vehicle dataset was obtained from **Kaggle** and contains four vehicle categories:

- Ambulance
- Bus
- Car
- Truck

The COCO8 dataset is used for the model-validation stage in Part 3.

---

# Requirements

- Python 3.x
- VS Code or Jupyter Notebook
- Ultralytics
- PyTorch
- OpenCV

Install Ultralytics:

```bash
pip install ultralytics
```

---

# How to Run

1. Open the project directory.
2. Install the required Python packages.
3. Open `lec1.ipynb` using VS Code or Jupyter Notebook.
4. Make sure the required input files and dataset paths are available.
5. Run the notebook cells in order.
6. For custom training, run Part 4.1 first to prepare `vehicle_dataset` and `vehicle.yaml`, then run Part 4.2.

---

# Project Structure

```text
SADAIA/
├── README.md
├── .gitignore
├── lec1.ipynb
├── bus.jpg
├── road_trafifc.mp4
├── vehicle.yaml
├── training_image/
└── vehicle_dataset/
```

---

# Technologies

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# Scope and Limitations

The project demonstrates object detection, instance segmentation, traffic object tracking, COCO8 model evaluation, custom vehicle dataset preparation, and custom vehicle model training.

The custom model focuses on four classes:

- Ambulance
- Bus
- Car
- Truck

Performance can vary depending on dataset quality, object size, lighting, occlusion, and the selected confidence and IoU thresholds.

---

# Conclusion

This project demonstrates how Ultralytics YOLO can be applied to multiple computer vision tasks, including image detection, instance segmentation, traffic object tracking, model evaluation, and custom vehicle model training.

The executed notebook contains the implementation and captured outputs for the completed project stages.
