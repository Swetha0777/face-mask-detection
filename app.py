"""
Streamlit Application for AI-Powered Face Mask Detection
Provides interactive multi-face image detection, live camera detection, model stats, and about page.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional AI UI Aesthetics
CUSTOM_CSS = """
<style>
    /* Global Styling */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    /* Cards & Containers */
    .css-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    .css-header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.4);
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-mask {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-nomask {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-neutral {
        background-color: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }

    /* Custom Titles */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* Flowchart Steps */
    .step-box {
        background: #1e293b;
        border-left: 4px solid #6366f1;
        padding: 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Lazy import src modules to handle environment checks smoothly
@st.cache_resource(show_spinner=False)
def load_app_modules():
    from src.model import load_mask_detector, build_mask_detector
    from src.detector import FaceMaskDetector
    from src.utils import get_label_badge, CLASS_NAMES
    return load_mask_detector, build_mask_detector, FaceMaskDetector, get_label_badge, CLASS_NAMES


load_mask_detector, build_mask_detector, FaceMaskDetector, get_label_badge, CLASS_NAMES = load_app_modules()

MODEL_PATH = "model/face_mask_detector.keras"


@st.cache_resource(show_spinner="Loading Deep Learning Model...")
def get_model():
    """Cached loader for Keras face mask detection model."""
    if os.path.exists(MODEL_PATH):
        return load_mask_detector(MODEL_PATH)
    return None


@st.cache_resource(show_spinner=False)
def get_detector(_model):
    """Cached detector instance."""
    return FaceMaskDetector(model=_model)


# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/isometric-reflection/100/medical-mask.png", width=70)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        [
            "🏠 Dashboard",
            "📷 Image Detection",
            "📸 Camera Detection",
            "📊 Model Information",
            "ℹ️ About"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("### System Status")
    
    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        st.markdown('<span class="metric-badge badge-mask">● Model Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="metric-badge badge-nomask">● Model Missing</span>', unsafe_allow_html=True)
        st.caption("Run `python train.py` to train model.")

    st.caption("Version 1.0.0 | Python 3.11+ | Keras")


# Header Banner
def render_header():
    st.markdown("""
    <div class="css-header-card">
        <h1 style="margin:0; font-size:2.4rem;">AI-Powered Face Mask Detection</h1>
        <p class="subtitle" style="margin-top:4px;">Real-time computer vision for detecting mask compliance using deep learning</p>
    </div>
    """, unsafe_allow_html=True)


# PAGE 1: DASHBOARD
if page == "🏠 Dashboard":
    render_header()

    model_obj = get_model()

    if not model_exists:
        st.warning("⚠️ Trained model `model/face_mask_detector.keras` was not found.")
        st.info("You can train the model anytime by executing `python train.py` in your terminal.")
        if st.button("⚡ Train & Build Model Now (Quick Demo Run)"):
            with st.spinner("Training baseline model... Please wait a few seconds..."):
                from train import train_model
                train_model(epochs=3, batch_size=16)
                st.cache_resource.clear()
                st.rerun()

    # Overview Metrics Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="css-card">
            <span style="color:#94a3b8; font-size:0.85rem;">SUPPORTED CLASSES</span>
            <h2 style="margin:4px 0; color:#818cf8 !important;">2 Classes</h2>
            <span style="color:#64748b; font-size:0.8rem;">Mask / No Mask</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="css-card">
            <span style="color:#94a3b8; font-size:0.85rem;">DETECTION TYPE</span>
            <h2 style="margin:4px 0; color:#38bdf8 !important;">Face Detection</h2>
            <span style="color:#64748b; font-size:0.8rem;">OpenCV Haar Cascade</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="css-card">
            <span style="color:#94a3b8; font-size:0.85rem;">MODEL ARCHITECTURE</span>
            <h2 style="margin:4px 0; color:#34d399 !important;">MobileNetV2</h2>
            <span style="color:#64748b; font-size:0.8rem;">Transfer Learning CNN</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="css-card">
            <span style="color:#94a3b8; font-size:0.85rem;">INPUT MODES</span>
            <h2 style="margin:4px 0; color:#f472b6 !important;">Image / Camera</h2>
            <span style="color:#64748b; font-size:0.8rem;">JPG, PNG, Webcam</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### How It Works")
    st.markdown("""
    <div class="css-card">
        <div class="step-box">
            <strong>Step 1: Image Capture / Upload</strong><br>
            <span style="color:#94a3b8; font-size:0.9rem;">Upload a target photograph or capture a live frame via webcam.</span>
        </div>
        <div class="step-box">
            <strong>Step 2: Face Localization</strong><br>
            <span style="color:#94a3b8; font-size:0.9rem;">OpenCV Haar Cascade detects bounding coordinates for all human faces.</span>
        </div>
        <div class="step-box">
            <strong>Step 3: Face Preprocessing</strong><br>
            <span style="color:#94a3b8; font-size:0.9rem;">Crops face ROI, resizes to 224x224 RGB, and normalizes pixel intensities to [-1, 1].</span>
        </div>
        <div class="step-box">
            <strong>Step 4: Deep Learning Inference</strong><br>
            <span style="color:#94a3b8; font-size:0.9rem;">Evaluates feature representations through MobileNetV2 classification head.</span>
        </div>
        <div class="step-box">
            <strong>Step 5: Visual Annotation & Analytics</strong><br>
            <span style="color:#94a3b8; font-size:0.9rem;">Renders green/red bounding boxes, label badges, and confidence metrics.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# PAGE 2: IMAGE DETECTION
elif page == "📷 Image Detection":
    render_header()
    st.markdown("### Analyze Photographs")
    st.caption("Upload single or group images to detect face mask compliance.")

    model_obj = get_model()
    detector = get_detector(model_obj)

    if model_obj is None:
        st.warning("⚠️ Model is not loaded yet. Predictions will use fallback bounds. Please train the model via `python train.py`.")

    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Original Upload")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### Analysis Result")
            with st.spinner("Processing face detection and model predictions..."):
                annotated_img, summary, details = detector.process_image(image)
                st.image(annotated_img, use_container_width=True)

        st.markdown("### Detection Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Faces Detected", summary["total_faces"])
        m2.metric("Wearing Mask 🟢", summary["with_mask"])
        m3.metric("Without Mask 🔴", summary["without_mask"])

        if len(details) > 0:
            st.markdown("### Detailed Face Classifications")
            for face in details:
                with st.container():
                    st.markdown('<div class="css-card">', unsafe_allow_html=True)
                    fc1, fc2, fc3 = st.columns([1, 2, 2])
                    with fc1:
                        st.image(face["face_crop"], width=110, caption=f"Face #{face['face_id']}")
                    with fc2:
                        badge_html = f'<span class="metric-badge badge-mask">{face["badge"]}</span>' if face["label"] == "with_mask" else f'<span class="metric-badge badge-nomask">{face["badge"]}</span>'
                        st.markdown(f"**Classification:** {badge_html}", unsafe_allow_html=True)
                        st.markdown(f"**Bounding Box (x,y,w,h):** `{face['bbox']}`")
                    with fc3:
                        st.markdown(f"**Confidence Level:** `{face['confidence']:.2f}%`")
                        st.progress(min(1.0, face["confidence"] / 100.0))
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No faces were detected in the uploaded image. Try an image with clearer frontal lighting.")


# PAGE 3: CAMERA DETECTION
elif page == "📸 Camera Detection":
    render_header()
    st.markdown("### Webcam Snapshot Detection")
    st.caption("Capture a live frame from your camera to evaluate mask compliance.")

    model_obj = get_model()
    detector = get_detector(model_obj)

    if model_obj is None:
        st.warning("⚠️ Model is not loaded. Train the model via `python train.py` first.")

    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        annotated_img, summary, details = detector.process_image(image)

        st.markdown("### Detection Output")
        col_cam1, col_cam2 = st.columns([1, 1])

        with col_cam1:
            st.image(annotated_img, caption="Analyzed Bounding Boxes", use_container_width=True)

        with col_cam2:
            st.markdown("#### Real-time Metrics")
            st.metric("Total Faces Detected", summary["total_faces"])
            st.metric("Mask Compliance 🟢", summary["with_mask"])
            st.metric("Non-Compliant 🔴", summary["without_mask"])

            if summary["total_faces"] > 0:
                compliance_rate = (summary["with_mask"] / summary["total_faces"]) * 100.0
                st.markdown(f"**Overall Compliance Rate:** `{compliance_rate:.1f}%`")
                st.progress(compliance_rate / 100.0)

        if len(details) > 0:
            st.markdown("#### Detected Faces Breakdown")
            for face in details:
                st.markdown(f"- **Face #{face['face_id']}:** {face['badge']} | Confidence: `{face['confidence']:.2f}%`")


# PAGE 4: MODEL INFORMATION
elif page == "📊 Model Information":
    render_header()
    st.markdown("### Deep Learning Architecture & Evaluation Metrics")

    metrics_path = "model/metrics.json"
    eval_path = "model/evaluation_report.json"
    history_plot = "model/training_history.png"
    cm_plot = "model/confusion_matrix.png"

    c_m1, c_m2 = st.columns(2)

    with c_m1:
        st.markdown("""
        <div class="css-card">
            <h3>Architecture Technical Specifications</h3>
            <ul>
                <li><strong>Base Model:</strong> MobileNetV2 (ImageNet Pre-trained)</li>
                <li><strong>Input Dimension:</strong> 224 × 224 × 3 RGB</li>
                <li><strong>Pooling Layer:</strong> AveragePooling2D (7×7)</li>
                <li><strong>Dense Layers:</strong> 128 units (ReLU) + Dropout (0.5)</li>
                <li><strong>Output Layer:</strong> 2 units (Softmax)</li>
                <li><strong>Optimizer:</strong> Adam (Learning Rate = 1e-4)</li>
                <li><strong>Loss Function:</strong> Categorical Crossentropy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c_m2:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)

            st.markdown("""
            <div class="css-card">
                <h3>Training Metrics Summary</h3>
            """, unsafe_allow_html=True)
            st.write(f"**Training Accuracy:** `{metrics_data.get('train_accuracy', 0.0) * 100:.2f}%`")
            st.write(f"**Validation Accuracy:** `{metrics_data.get('val_accuracy', 0.0) * 100:.2f}%`")
            st.write(f"**Training Loss:** `{metrics_data.get('train_loss', 0.0):.4f}`")
            st.write(f"**Validation Loss:** `{metrics_data.get('val_loss', 0.0):.4f}`")
            st.write(f"**Epochs Trained:** `{metrics_data.get('epochs_completed', 0)}`")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Training metrics file `model/metrics.json` not found. Run `python train.py` to generate.")

    st.markdown("---")
    st.markdown("### Evaluation Visualizations")

    v1, v2 = st.columns(2)

    with v1:
        st.markdown("#### Training & Validation History")
        if os.path.exists(history_plot):
            st.image(history_plot, use_container_width=True)
        else:
            st.caption("Training history plot will be generated after executing `python train.py`.")

    with v2:
        st.markdown("#### Confusion Matrix Heatmap")
        if os.path.exists(cm_plot):
            st.image(cm_plot, use_container_width=True)
        else:
            st.caption("Confusion matrix plot will be generated after executing `python evaluate.py`.")


# PAGE 5: ABOUT
elif page == "ℹ️ About":
    render_header()
    st.markdown("""
    <div class="css-card">
        <h2>About the Project</h2>
        <p><strong>Face Mask Detection using Machine Learning</strong> is an AI-powered computer vision system designed to automatically detect and verify whether individuals are wearing protective face masks in public or restricted settings.</p>

        <h3>1. Problem Statement</h3>
        <p>During global health crises and airborne pathogen outbreaks, face mask compliance is critical for minimizing transmission. Manual verification at building entry points is labor-intensive, slow, and prone to human error. An automated vision pipeline provides scalable, instant compliance monitoring.</p>

        <h3>2. Technical Approach</h3>
        <ul>
            <li><strong>Face Localization:</strong> Employs OpenCV Haar Cascade frontal face detection algorithm to identify bounding box coordinates of all faces within an input frame.</li>
            <li><strong>Deep Learning Classification:</strong> Uses MobileNetV2 transfer learning architecture pre-trained on ImageNet, fine-tuned for binary face mask classification.</li>
            <li><strong>Normalization & Data Augmentation:</strong> Input images are resized to 224×224 and normalized. Augmentation techniques including rotation, shift, zoom, and horizontal flip are applied during training to prevent overfitting.</li>
        </ul>

        <h3>3. Real-World Applications</h3>
        <ul>
            <li>Airports and Transportation Hubs</li>
            <li>Hospitals and Healthcare Facilities</li>
            <li>Corporate Offices & Factory Entrances</li>
            <li>Educational Institutions</li>
        </ul>

        <h3>4. Limitations & Edge Cases</h3>
        <ul>
            <li>Extreme head poses or severe side profile angles may escape Haar Cascade detection.</li>
            <li>Poor lighting conditions or harsh shadows can reduce classification confidence.</li>
            <li>Hand occlusions or scarves covering the face may yield false positives/negatives.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
