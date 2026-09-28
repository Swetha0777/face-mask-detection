# Face Mask Detection using Machine Learning 😷 AI Computer Vision System

An AI-powered computer vision and deep learning application built with **Python**, **TensorFlow/Keras (MobileNetV2)**, **OpenCV**, and **Streamlit** to detect human faces in images or webcam feeds and classify whether each person is **Wearing a Mask (🟢 MASK)** or **Not Wearing a Mask (🔴 NO MASK)**.

---

## 📌 1. Problem Statement

During public health emergencies, airborne epidemic outbreaks, and cleanroom operating standards, verifying face mask compliance is vital for public safety. Manual monitoring at entrances, transport hubs, and facilities is labor-intensive, inefficient, and susceptible to human oversight.

This project delivers an automated, scalable machine learning system that performs instant multi-face localization and high-accuracy mask classification in real time.

---

## 🎯 2. Project Objective

- Detect single or multiple human faces in an uploaded image or live webcam feed using OpenCV.
- Classify each detected face into one of two classes:
  1. `with_mask` (Wearing a Mask)
  2. `without_mask` (Not Wearing a Mask)
- Provide a modern, interactive **Streamlit** web application for real-time inference, dataset training, model evaluation metrics, and technical insights.

---

## ✨ 3. Core Features

- **🏠 Interactive Dashboard**: Live project analytics overview, architecture specs, and step-by-step visual workflow.
- **📷 Image Detection**: Drag-and-drop support for `JPG`, `JPEG`, and `PNG` images with multi-face detection, bounding boxes, status badges, and confidence metrics.
- **📸 Camera Detection**: Live webcam capture via Streamlit's native camera input for instant compliance testing.
- **📊 Model Information**: Detailed deep learning metrics including training/validation loss/accuracy curves, confusion matrix, and precision/recall stats.
- **ℹ️ About Page**: Comprehensive system documentation covering machine learning methodology, real-world deployment cases, and system limitations.
- **⚡ Automated Synthetic Fallback**: `train.py` automatically generates synthetic sample face data if raw datasets are absent, guaranteeing smooth zero-setup execution.

---

## 🛠️ 4. Tech Stack

- **Language**: Python 3.11+
- **Deep Learning**: TensorFlow 2.x / Keras (MobileNetV2 Transfer Learning)
- **Computer Vision**: OpenCV (`opencv-python`)
- **Web UI**: Streamlit
- **Data Processing**: NumPy, Pandas, Pillow (PIL)
- **Metrics & Visualization**: Scikit-Learn, Matplotlib

---

## 📁 5. Project Structure

```text
facemask/
│
├── app.py                      # Main Streamlit Web Application
├── train.py                    # Model Training Script
├── evaluate.py                 # Evaluation & Metrics Script
├── requirements.txt            # Project Dependencies
├── README.md                   # Complete Documentation
│
├── model/                      # Saved Models & Artifacts
│   ├── README.md
│   ├── face_mask_detector.keras (Generated after training)
│   ├── metrics.json            (Generated after training)
│   ├── training_history.png    (Generated after training)
│   ├── evaluation_report.json  (Generated after evaluation)
│   └── confusion_matrix.png    (Generated after evaluation)
│
├── src/                        # Source Code Modules
│   ├── __init__.py
│   ├── model.py                # MobileNetV2 & CNN Architecture
│   ├── detector.py             # OpenCV Face Detector & Classifier
│   └── utils.py                # Preprocessing & Plotting Helpers
│
├── data/                       # Training Dataset Directory
│   ├── README.md
│   ├── with_mask/              # Images of people wearing face masks
│   └── without_mask/           # Images of people NOT wearing face masks
│
└── assets/                     # Visual Assets & Screenshots
    └── README.md
```

---

## 📂 6. Dataset Format

Organize your dataset into two subfolders inside `data/`:

```text
data/
├── with_mask/
│   ├── face1.jpg
│   ├── face2.png
│   └── ...
└── without_mask/
    ├── face1.jpg
    ├── face2.png
    └── ...
```

> **Note**: If `data/with_mask` or `data/without_mask` are empty when you run `python train.py`, the training script will automatically create synthetic face sample images so you can test the entire pipeline right out of the box!

---

## 📥 7. Installation & Setup

1. **Clone or Navigate to Project Directory**:
   ```bash
   cd facemask
   ```

2. **(Optional) Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 8. Model Training Instructions

To train the MobileNetV2 Deep Learning Model on your dataset:

```bash
python train.py --epochs 15 --batch_size 32 --lr 0.0001
```

### What `train.py` does:
1. Loads dataset images from `data/with_mask` and `data/without_mask`.
2. Applies data augmentation (rotations, shifts, zooms, flips).
3. Fine-tunes MobileNetV2 classification head using Adam optimizer.
4. Saves the best model to `model/face_mask_detector.keras`.
5. Exports training accuracy/loss plots to `model/training_history.png` and summary metrics to `model/metrics.json`.

---

## 📈 9. Model Evaluation Instructions

To calculate accuracy, precision, recall, F1-score, and confusion matrix:

```bash
python evaluate.py
```

### What `evaluate.py` does:
1. Loads `model/face_mask_detector.keras`.
2. Evaluates predictions on validation/test images.
3. Generates confusion matrix heatmap saved to `model/confusion_matrix.png`.
4. Saves comprehensive classification report to `model/evaluation_report.json`.

---

## 💻 10. Running the Streamlit Web Application

Launch the Streamlit web app with:

```bash
streamlit run app.py
```

Open your web browser at `http://localhost:8501`.

---

## 📊 11. Expected Output

- **Green Bounding Box & Label `🟢 MASK 98.5%`** around faces detected with a mask.
- **Red Bounding Box & Label `🔴 NO MASK 96.2%`** around faces detected without a mask.
- Summary analytics box displaying:
  - **Total Faces Detected**
  - **Wearing Mask Count**
  - **Without Mask Count**

---

## 🔮 12. Future Enhancements

- Integrate **MediaPipe Face Mesh** or **YOLOv8-Face** for higher resilience to side profiles and head tilts.
- Add live video stream processing via RTSP/IP camera input (`streamlit-webrtc`).
- Multi-class expansion (e.g. `incorrectly_worn_mask`).
- Audio alerts / alarm trigger when unmasked individuals are detected.
