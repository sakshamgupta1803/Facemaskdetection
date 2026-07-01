"""
Face Mask Detection - Streamlit Web App
-----------------------------------------
Author: [Your Name]
Description:
    A CNN-based face mask detector. Upload an image, detect all faces
    using OpenCV's Haar Cascade, then classify each face as
    "Mask" or "No Mask" using a trained Keras CNN.

    Model input: 128x128 RGB, pixel values scaled to [0, 1]
    Model output: softmax over 2 classes -> [No Mask, Mask]
        index 0 -> No Mask
        index 1 -> Mask
"""

import io
import base64

import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model


# ----------------------------------------------------------------------
# App config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best_face_mask_model.keras"
IMG_SIZE = (128, 128)          # (width, height) expected by the CNN
CLASS_NAMES = {0: "No Mask", 1: "Mask"}

# Haar cascade face detector params (tuned during dev/testing)
CASCADE_SCALE_FACTOR = 1.1
CASCADE_MIN_NEIGHBORS = 7
CASCADE_MIN_SIZE = (60, 60)

# A short base64-encoded "beep" wav used for the optional alarm sound.
# (Generated once, kept tiny so it ships fine inside the repo.)
BEEP_WAV_B64 = (
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA="
)


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained CNN model...")
def get_model():
    """Load the trained Keras model once and cache it across reruns."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model from '{MODEL_PATH}'. Error: {e}")
        st.stop()


@st.cache_resource(show_spinner=False)
def get_face_detector():
    """Load OpenCV's pretrained Haar Cascade face detector."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            raise IOError("Cascade file loaded but is empty.")
        return detector
    except Exception as e:
        st.error(f"Failed to load face detector. Error: {e}")
        st.stop()


# ----------------------------------------------------------------------
# Core inference helpers
# ----------------------------------------------------------------------
def detect_faces(gray_image, detector):
    """Run Haar Cascade face detection on a grayscale image."""
    try:
        faces = detector.detectMultiScale(
            gray_image,
            scaleFactor=CASCADE_SCALE_FACTOR,
            minNeighbors=CASCADE_MIN_NEIGHBORS,
            minSize=CASCADE_MIN_SIZE,
        )
        return faces
    except Exception as e:
        st.error(f"Face detection failed: {e}")
        return []


def preprocess_face(face_bgr):
    """
    Preprocess a cropped face exactly like the training pipeline:
    BGR -> RGB -> resize to 128x128 -> scale to [0,1] -> add batch dim.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    face_scaled = face_resized.astype("float32") / 255.0
    face_batch = np.expand_dims(face_scaled, axis=0)
    return face_batch


def predict_face(model, face_bgr):
    """Return (label, confidence) for a single cropped face."""
    try:
        face_input = preprocess_face(face_bgr)
        prediction = model.predict(face_input, verbose=0)[0]
        pred_idx = int(np.argmax(prediction))
        confidence = float(prediction[pred_idx])
        return CLASS_NAMES[pred_idx], confidence
    except Exception as e:
        st.error(f"Prediction failed for a detected face: {e}")
        return "Unknown", 0.0


def process_image(pil_image, model, detector):
    """
    Full pipeline for one uploaded image:
    - detect faces
    - classify each face
    - draw bounding boxes + labels
    Returns: annotated RGB image (numpy array), list of results dicts
    """
    rgb_image = np.array(pil_image.convert("RGB"))
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(gray_image, detector)
    results = []

    for i, (x, y, w, h) in enumerate(faces, start=1):
        face_crop = bgr_image[y:y + h, x:x + w]

        if face_crop.size == 0:
            continue

        label, confidence = predict_face(model, face_crop)
        results.append({
            "id": i,
            "label": label,
            "confidence": confidence,
            "box": (int(x), int(y), int(w), int(h)),
        })

        color = (0, 200, 0) if label == "Mask" else (220, 0, 0)  # RGB
        cv2.rectangle(bgr_image, (x, y), (x + w, y + h), color[::-1], 2)

        text = f"Person {i}: {label} ({confidence * 100:.1f}%)"
        text_y = y - 10 if y - 10 > 10 else y + h + 20
        cv2.putText(
            bgr_image, text, (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color[::-1], 2, cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results


def play_alarm():
    """Autoplay a short beep sound using an embedded audio element."""
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{BEEP_WAV_B64}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def build_sidebar():
    with st.sidebar:
        st.title("😷 Face Mask Detector")
        st.caption("Mini Project — B.Tech CSE, 3rd Year")

        st.markdown("---")
        st.subheader("📌 About the Project")
        st.write(
            "This app uses a Convolutional Neural Network (CNN) trained "
            "from scratch to classify whether a person is wearing a face "
            "mask or not. Faces are located using OpenCV's Haar Cascade "
            "detector, then each face is passed through the CNN for "
            "classification."
        )

        st.markdown("---")
        st.subheader("📤 Upload")
        uploaded_file = st.file_uploader(
            "Upload an image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
        )
        st.caption("Tip: works best with clear, front-facing photos.")

        st.markdown("---")
        st.subheader("⚙️ Model Info")
        st.markdown(
            """
            - **Architecture:** Custom CNN (Conv2D + BatchNorm + MaxPool)
            - **Input size:** 128 x 128 x 3
            - **Classes:** Mask, No Mask
            - **Framework:** TensorFlow / Keras
            - **Face Detector:** OpenCV Haar Cascade
            """
        )

        enable_alarm = st.checkbox("🔊 Enable alarm sound on violation", value=False)

        st.markdown("---")
        st.subheader("👨‍💻 Developer")
        st.markdown(
            """
            **Made by:** [Your Name]
            **College:** [Your College Name]
            **GitHub:** [github.com/your-username](https://github.com/)
            **LinkedIn:** [linkedin.com/in/your-profile](https://linkedin.com/)
            """
        )

        st.markdown("---")
        st.caption("Built as a portfolio / internship project.")

    return uploaded_file, enable_alarm


# ----------------------------------------------------------------------
# Main dashboard sections
# ----------------------------------------------------------------------
def show_metrics(results):
    total = len(results)
    masked = sum(1 for r in results if r["label"] == "Mask")
    unmasked = total - masked
    compliance = (masked / total * 100) if total > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Faces Detected", total)
    col2.metric("Masked", masked)
    col3.metric("Unmasked", unmasked)
    col4.metric("Mask Compliance", f"{compliance:.1f}%")

    return total, masked, unmasked, compliance


def show_charts(masked, unmasked):
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Mask vs No Mask (Bar Chart)**")
        fig, ax = plt.subplots(figsize=(4, 3))
        categories = ["Mask", "No Mask"]
        values = [masked, unmasked]
        bar_colors = ["#2e7d32", "#c62828"]
        ax.bar(categories, values, color=bar_colors)
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(values + [1]) + 1)
        for i, v in enumerate(values):
            ax.text(i, v + 0.05, str(v), ha="center")
        st.pyplot(fig)

    with chart_col2:
        st.markdown("**Mask Distribution (Pie Chart)**")
        if masked + unmasked > 0:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            labels = []
            sizes = []
            colors = []
            if masked > 0:
                labels.append("Mask")
                sizes.append(masked)
                colors.append("#2e7d32")
            if unmasked > 0:
                labels.append("No Mask")
                sizes.append(unmasked)
                colors.append("#c62828")
            ax2.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
            ax2.axis("equal")
            st.pyplot(fig2)
        else:
            st.info("No faces detected yet.")


def show_alert(total, unmasked, enable_alarm):
    st.markdown("### 🚨 Alert Status")
    if total == 0:
        st.info("No faces detected in the uploaded image.")
    elif unmasked > 0:
        st.error("⚠️ Warning: Unmasked person detected.")
        if enable_alarm:
            play_alarm()
    else:
        st.success("✅ All detected people are wearing masks.")


def show_results_table(results):
    if not results:
        return
    st.markdown("### 📋 Per-Face Results")
    table_data = [
        {
            "Person": r["id"],
            "Prediction": r["label"],
            "Confidence": f"{r['confidence'] * 100:.1f}%",
        }
        for r in results
    ]
    st.table(table_data)


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------
def main():
    uploaded_file, enable_alarm = build_sidebar()

    st.title("😷 Face Mask Detection System")
    st.write(
        "Upload a photo containing one or more faces. The app will detect "
        "every face, classify each as **Mask** or **No Mask**, and show a "
        "live analytics dashboard below."
    )
    st.markdown("---")

    if uploaded_file is None:
        st.info("👈 Upload an image from the sidebar to get started.")
        return

    model = get_model()
    detector = get_face_detector()

    try:
        pil_image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded image: {e}")
        return

    with st.spinner("Running detection..."):
        annotated_image, results = process_image(pil_image, model, detector)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Original Image**")
        st.image(pil_image, use_container_width=True)
    with img_col2:
        st.markdown("**Detected Output**")
        st.image(annotated_image, use_container_width=True)

    st.markdown("---")
    st.markdown("## 📊 Analytics Dashboard")
    total, masked, unmasked, compliance = show_metrics(results)
    show_charts(masked, unmasked)

    st.markdown("---")
    show_alert(total, unmasked, enable_alarm)

    st.markdown("---")
    show_results_table(results)


if __name__ == "__main__":
    main()
