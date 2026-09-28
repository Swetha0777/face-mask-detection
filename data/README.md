# Data Directory

This directory contains the dataset used for training and evaluating the **Face Mask Detection** model.

## Folder Structure

```text
data/
├── with_mask/       # Place images of people wearing face masks here
└── without_mask/    # Place images of people NOT wearing face masks here
```

## Dataset Specifications

- **Format**: JPG, JPEG, PNG, or WEBP.
- **Recommended Count**: Minimum 100–500 images per class for custom CNNs, or 500+ images per class for transfer learning (MobileNetV2).
- **Target Face Alignment**: Clear frontal face shots yield highest detection accuracy.

## Dataset Sources

Popular open datasets for Face Mask Detection:
1. **Kaggle Face Mask Dataset** (e.g. Prajna Bhandary's Face Mask Dataset)
2. **PyImageSearch Face Mask Dataset**
3. **Real-World Masked Face Dataset (RMFD)**

## Synthetic Data Generation

If `train.py` is executed when these folders are empty, `train.py` can automatically generate synthetic sample face data for initial testing.
