"""
Face Detection and Mask Classification Pipeline
Combines OpenCV Haar Cascade Face Detection with Keras Mask Classification Model.
"""

import cv2
import numpy as np
from PIL import Image
from src.utils import preprocess_face, CLASS_NAMES, get_label_badge


class FaceMaskDetector:
    """
    Orchestrates face detection using OpenCV Haar Cascade and classification using Keras model.
    """

    def __init__(self, model=None, cascade_path=None):
        """
        Initializes FaceMaskDetector with model instance and Haar Cascade classifier.
        """
        self.model = model
        
        # Load OpenCV default frontal face Haar cascade
        if cascade_path and cv2.os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            default_cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(default_cascade)

        if self.face_cascade.empty():
            print("[WARNING] Could not load OpenCV Haar Cascade frontal face classifier!")

    def set_model(self, model):
        """Update active machine learning model."""
        self.model = model

    def process_image(self, image_input):
        """
        Detect faces in image, run mask classification model, draw bounding boxes and annotations.
        
        Args:
            image_input (PIL.Image.Image or np.ndarray): Input image.
            
        Returns:
            tuple: (annotated_rgb_image, summary_dict, face_details_list)
        """
        # Ensure image is in OpenCV BGR format for face detector
        if isinstance(image_input, Image.Image):
            rgb_orig = np.array(image_input.convert("RGB"))
            bgr_img = cv2.cvtColor(rgb_orig, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Assume standard RGB input if passed as numpy
                rgb_orig = image_input.copy()
                bgr_img = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
            else:
                rgb_orig = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                bgr_img = image_input.copy()
        else:
            raise ValueError("Unsupported image input format.")

        annotated_bgr = bgr_img.copy()
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # Equalize histogram for improved detection under variable lighting
        gray_eq = cv2.equalizeHist(gray)

        # Detect faces in gray image
        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_details = []
        with_mask_count = 0
        without_mask_count = 0

        img_h, img_w = bgr_img.shape[:2]

        for idx, (x, y, w, h) in enumerate(faces):
            # Expand bounding box slightly (10% padding) to capture full face context
            padding = int(max(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_w, x + w + padding)
            y2 = min(img_h, y + h + padding)

            face_roi_bgr = bgr_img[y1:y2, x1:x2]

            if face_roi_bgr.size == 0:
                continue

            # Run Deep Learning Prediction if model is loaded
            if self.model is not None:
                tensor = preprocess_face(face_roi_bgr)
                preds = self.model.predict(tensor, verbose=0)
                pred_idx = np.argmax(preds[0])
                confidence = float(preds[0][pred_idx] * 100.0)
                predicted_label = CLASS_NAMES[pred_idx]
            else:
                predicted_label = "without_mask"
                confidence = 0.0

            # Increment count
            if predicted_label == "with_mask":
                with_mask_count += 1
                color = (16, 185, 129)  # Green (BGR: 129, 185, 16) -> (16, 185, 129) in BGR
                box_color_bgr = (129, 185, 16)
                badge_text, _, badge_color_hex = get_label_badge("with_mask")
            else:
                without_mask_count += 1
                box_color_bgr = (68, 68, 239)  # Red BGR
                badge_text, _, badge_color_hex = get_label_badge("without_mask")

            # Draw bounding box on face
            thickness = max(2, int(min(img_w, img_h) / 200))
            cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), box_color_bgr, thickness)

            # Draw label banner above bounding box
            display_text = f"{'MASK' if predicted_label == 'with_mask' else 'NO MASK'} {confidence:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, w / 220.0)
            text_thickness = max(1, int(font_scale * 2))

            (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, text_thickness)
            
            # Position box above face if space permits
            text_y1 = max(0, y - text_h - 10)
            text_y2 = max(text_h + 10, y)

            cv2.rectangle(annotated_bgr, (x, text_y1), (x + text_w + 12, text_y2), box_color_bgr, -1)
            cv2.putText(
                annotated_bgr,
                display_text,
                (x + 6, text_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA
            )

            # Store face details and RGB crop for UI detailed card rendering
            face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
            face_details.append({
                "face_id": idx + 1,
                "bbox": (x, y, w, h),
                "label": predicted_label,
                "badge": badge_text,
                "color_hex": badge_color_hex,
                "confidence": confidence,
                "face_crop": face_roi_rgb
            })

        # Convert final annotated image back to RGB for PIL / Streamlit
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        summary_dict = {
            "total_faces": len(faces),
            "with_mask": with_mask_count,
            "without_mask": without_mask_count
        }

        return annotated_rgb, summary_dict, face_details
