# YOLO26n — Smart Traffic Object Detection (100 Epochs)

A fine-tuned **YOLO26n** object-detection model for identifying eight categories of road and emergency vehicles. This repository contains the trained model weights and the training/evaluation artifacts.

## 1. Supported classes

The model uses the following class order:

| Class ID | Class name |
|---:|---|
| 0 | Ambulance |
| 1 | Bus |
| 2 | Car |
| 3 | Fire engine |
| 4 | Military |
| 5 | Motorcycle |
| 6 | Police car |
| 7 | Truck |

## 2. Repository contents

```text
.
├── weights/
│   ├── best.pt
│   └── last.pt
├── args.yaml
├── results.csv
├── results.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── BoxP_curve.png
└── BoxR_curve.png
```

- **`weights/best.pt`** — recommended checkpoint for inference and application integration.
- **`weights/last.pt`** — final checkpoint saved at the end of training.
- **`results.csv`** — epoch-by-epoch training and validation metrics.
- **Evaluation plots** — learning curves, precision-recall curve, and confusion matrices.
- **`args.yaml`** — recorded training configuration.

The dataset itself is not included in this repository.

## 3. Dataset

The training configuration references an eight-class YOLO-format dataset.

| Split | Images |
|---|---:|
| Train | 6,077 |
| Validation | 1,519 |
| **Total** | **7,596** |

The validation set contains 3,679 labeled object instances. Empty label files were retained as negative images where no target object was annotated.

Dataset configuration used during training:

- Image size: 640 × 640
- Batch size: 16
- Number of classes: 8
- Validation split: `valid`

The original `data.yaml` and image dataset are not bundled here. Update the dataset paths in your own `data.yaml` before retraining.

## 4. Training configuration

Selected settings recorded in `args.yaml`:

| Parameter | Value |
|---|---|
| Model | YOLO26n |
| Task | Object detection |
| Epoch target | 100 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 0.000833 |
| Seed | 42 |
| Mixed precision (AMP) | Enabled |
| Ultralytics | 8.4.194 |
| Training GPU | NVIDIA Tesla T4 |

Training was continued from a previous checkpoint using the `last.pt` resume workflow. The CSV contains 100 epoch records, numbered 1 through 100.

## 5. Evaluation results

### Best-checkpoint validation

The validation summary for `best.pt` reported:

| Metric | Result |
|---|---:|
| Precision | 87.3% |
| Recall | 83.3% |
| mAP@50 | 89.5% |
| mAP@50–95 | 72.1% |

### Epoch 100 record in `results.csv`

| Metric | Result |
|---|---:|
| Precision | 88.84% |
| Recall | 83.10% |
| mAP@50 | 89.31% |
| mAP@50–95 | 71.69% |

The highest mAP@50–95 recorded in the CSV was **72.18% at epoch 72**. Individual metric peaks can occur at different epochs; therefore, the final epoch is not necessarily the best checkpoint.

### Per-class validation summary for best.pt

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Ambulance | 193 | 225 | 97.6% | 91.1% | 97.8% | 89.0% |
| Bus | 176 | 210 | 88.2% | 71.4% | 85.3% | 66.5% |
| Car | 664 | 1,760 | 84.2% | 81.4% | 87.2% | 67.4% |
| Fire engine | 158 | 207 | 92.7% | 89.9% | 94.8% | 81.2% |
| Military | 14 | 20 | 71.8% | 90.0% | 88.1% | 65.6% |
| Motorcycle | 175 | 300 | 90.1% | 79.3% | 88.2% | 60.4% |
| Police car | 359 | 547 | 89.7% | 89.2% | 93.1% | 80.8% |
| Truck | 274 | 410 | 83.9% | 73.9% | 81.8% | 66.3% |

**Interpretation:** Ambulance, Fire engine, and Police car achieved the strongest overall class results. Bus and Truck have comparatively lower recall, while Motorcycle has a larger gap between mAP@50 and mAP@50–95, suggesting localization quality deserves further inspection. The Military class has very few validation examples, so its metrics should be interpreted cautiously.

Inference speed during validation on the Tesla T4 was approximately 6.4 ms per image. Actual application latency may differ depending on hardware, image processing, and GUI overhead.

## 6. Run inference

Install the required packages in a Python environment:

```bash
pip install ultralytics
```

### Python

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")

results = model.predict(
    source="path/to/image.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)

for result in results:
    print(result.boxes)
```

Replace `path/to/image.jpg` with the path to an image or video. For a webcam, use `source=0`.

### Command line

```bash
yolo detect predict model=weights/best.pt source="path/to/image.jpg" imgsz=640
```

## 7. Notes and limitations

- The reported metrics are validation results, not a guarantee of performance in every real-world traffic scene.
- Performance can vary with lighting, camera angle, distance, occlusion, weather, and camera quality.
- The Military class has limited validation data; additional representative data would improve confidence in its evaluation.
- Test the model on representative target-camera footage before deploying it in a traffic-control application.
- This repository provides an object detector. It does not, by itself, implement traffic-light phase control or guarantee safe traffic decisions.

## 8. License and dataset attribution

No dataset license or model redistribution license is specified in the training artifacts. Confirm the dataset's terms and add the appropriate license and attribution before redistributing this model or using it commercially.
