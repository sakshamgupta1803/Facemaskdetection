# 😷 Face Mask Detection System

A real-time-capable face mask detection web app built with a custom-trained
Convolutional Neural Network (CNN), OpenCV face detection, and Streamlit.

This project was built as a mini-project / portfolio piece to demonstrate
an end-to-end computer vision pipeline: dataset preparation → CNN training →
model evaluation → deployment as an interactive web app.

---

## 📌 Project Overview

The app takes an uploaded image, detects every face in it using OpenCV's
Haar Cascade classifier, and passes each detected face through a trained
CNN to classify it as **Mask** or **No Mask**. Results are shown with
bounding boxes on the image, plus a small analytics dashboard summarizing
mask compliance.

---

## ✨ Features

- 📤 Upload any JPG/PNG image
- 🧠 CNN-based Mask / No Mask classification
- 👥 **Multiple face detection** — every face in the image is detected and
  classified independently
- 🟩 🟥 Bounding boxes: green for Mask, red for No Mask
- 📊 **Analytics dashboard**:
  - Total faces detected
  - Masked / Unmasked counts
  - Mask compliance percentage
  - Bar chart + pie chart breakdown
- 🚨 **Alert system**: warns if any unmasked person is found, with an
  optional alarm sound
- 🖥️ Clean, minimal Streamlit UI

---

## 🏗️ Architecture

```
Uploaded Image
     │
     ▼
OpenCV Haar Cascade (face detection)
     │
     ▼
For each detected face:
   - Crop face region
   - Resize to 128x128
   - Normalize pixels to [0, 1]
     │
     ▼
CNN Model (best_face_mask_model.keras)
   - Conv2D + BatchNorm + MaxPool (x3 blocks)
   - Dense(128) + Dropout
   - Dense(2, softmax) → [No Mask, Mask]
     │
     ▼
Draw bounding box + label on image
     │
     ▼
Aggregate results → Analytics Dashboard + Alerts
```

**Model details**

| Property        | Value                          |
|------------------|---------------------------------|
| Input shape      | 128 x 128 x 3 (RGB)             |
| Preprocessing    | Resize → scale pixels to [0, 1] |
| Output           | Softmax over 2 classes          |
| Class 0          | No Mask                         |
| Class 1          | Mask                            |
| Framework        | TensorFlow / Keras              |
| Face detector    | OpenCV Haar Cascade (frontal face) |

---

## 🖼️ Screenshots

> _Add screenshots here after running the app locally._

- `assets/screenshot_upload.png` — Upload screen
- `assets/screenshot_detection.png` — Detection output with bounding boxes
- `assets/screenshot_dashboard.png` — Analytics dashboard

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure `best_face_mask_model.keras` is present in the project root
   (same folder as `app.py`).

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually
`http://localhost:8501`) in your browser.

1. Upload an image containing one or more faces from the sidebar.
2. View the annotated output with bounding boxes.
3. Check the analytics dashboard for mask compliance statistics.
4. If an unmasked face is detected, an alert (and optional alarm sound)
   is shown.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this project to a public GitHub repository, including:
   - `app.py`
   - `requirements.txt`
   - `best_face_mask_model.keras`
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with
   GitHub.
3. Click **"New app"**, select your repository, branch, and set the main
   file path to `app.py`.
4. Click **"Deploy"**. Streamlit Cloud will install dependencies from
   `requirements.txt` and launch the app automatically.

No additional configuration is required — the app loads the model relative
to the repository root.

---

## 📁 Folder Structure

```
face-mask-detection/
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Files/folders excluded from git
├── best_face_mask_model.keras   # Trained CNN model
└── assets/                      # Screenshots / static assets
```

---

## 🚀 Future Improvements

- [ ] Real-time webcam detection directly inside the Streamlit app (via
      `streamlit-webrtc`)
- [ ] Support for video file uploads
- [ ] Improve face detection with a DNN-based detector (e.g., MTCNN or
      OpenCV's DNN face detector) for better accuracy on angled faces
- [ ] Add mask-type classification (surgical / cloth / N95)
- [ ] Deploy a lightweight version using TensorFlow Lite for edge devices
- [ ] Add user authentication + detection history/logging
- [ ] Dockerize the app for easier deployment

---

## 👨‍💻 Developer

- **Name:** [Your Name]
- **College:** [Your College Name], 3rd Year, CSE
- **GitHub:** [github.com/your-username](https://github.com/)
- **LinkedIn:** [linkedin.com/in/your-profile](https://linkedin.com/)

---

## 📄 License

This project is open-source and available for educational use.
