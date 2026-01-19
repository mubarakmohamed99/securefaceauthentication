# Secure Facial Recognition & Liveness Detection

This project implements a secure identity verification system that combines **Active Liveness Detection** with **Face Recognition**.

It prevents spoofing attacks (using photos or videos) by requiring the user to perform randomized physical actions (Head turns, Mouth opening) before attempting to verify their identity against a pre-recorded database.

## 🌟 Features

* **Active Liveness Detection:** Users must pass a challenge-response sequence (Look Left, Look Right, Open Mouth, Nod Up) tracked via MediaPipe Face Mesh.
* **High-Accuracy Recognition:** Uses **InsightFace** (ArcFace) to generate 512D face embeddings for robust identity verification.
* **Interactive UI:** A polished OpenCV-based user interface with real-time feedback, progress indicators, and visual overlays.
* **Smart Stabilization:** The system stabilizes after the liveness check to ensure a clear, blur-free image is captured for identification.

## 🛠️ Installation

### 1. Prerequisites

* Python 3.11 is recommended (InsightFace and ONNX runtimes can be picky with newer Python versions).
* A working webcam.

### 2. Install Dependencies

This project uses a specific combination of libraries to prevent conflicts between `mediapipe` and `onnx` regarding Protocol Buffers (`protobuf`).

Run the following command to install the "Peacemaker" configuration:

```bash
pip install -r requirements.txt

```

**Note:** If you encounter errors regarding `protobuf`, ensure you do not have other versions installed globally, or run this in a virtual environment.

---

## 📂 Project Structure

```text
.
├── faces_db/               # (Create this) Place raw images of users here
│   ├── John_Doe/           # Folder name = User Name
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── Jane_Smith/
│       └── selfy.png
├── face_embeddings/        # (Auto-generated) Stores processed face vectors
├── extract_embeddings.py   # Script to process raw images into embeddings
├── main.py                 # The main application script
└── requirements.txt        # Dependency list

```

---

## 🚀 Usage Guide

### Step 1: Create the Face Database

Before the system can recognize anyone, you must register them.

1. Create a folder named `faces_db` in the project root.
2. Inside `faces_db`, create a subfolder for each person you want to recognize. The **folder name** will be used as their display name (e.g., `faces_db/Elon_Musk`).
3. Add clear photos of that person inside their folder.
* *Tip:* Use 3-5 different photos with good lighting for the best accuracy.



### Step 2: Generate Embeddings

Run the extraction script to convert the images into mathematical embeddings.

```bash
python extract_embeddings.py

```

* This will create a `face_embeddings/` folder.
* It generates `.pkl` files for each user.
* If you add new users later, just run this script again; it skips already processed users.

### Step 3: Run the Verification System

Start the main application:

```bash
python main.py

```

If you are running locally and want full **liveness + recognition**, install the full stack:

```bash
pip install -r requirements-full.txt

```

### Optional: Run the Streamlit Web UI

After installing the dependencies, you can launch a simple web interface (camera snapshot or image upload) with:

```bash
streamlit run streamlit_app.py

```
This UI reuses the same embeddings generated in `face_embeddings/` to perform identity verification from a single image. For full active liveness detection with real-time challenges, use the OpenCV application via `python main.py`.

On Streamlit Cloud, only the lighter dependencies from `requirements.txt` are installed (no MediaPipe). Liveness is therefore **disabled** in the cloud UI, but identity recognition still works using the precomputed embeddings and InsightFace.

### The Verification Flow

1. **Liveness Check:** The system will instruct you to perform actions (e.g., "Turn Head LEFT", "Open MOUTH").
2. **Stabilization:** Once liveness is verified, the UI turns green. Hold still for a moment.
3. **Identity Verification:** The system compares your face to the database.
* **Success:** Displays "IDENTITY VERIFIED" and your name.
* **Failure:** Continues scanning or stays in the verifying state if the match score is too low.



---

## ⚙️ Configuration

You can tweak the sensitivity of the system by modifying the constants at the top of `main.py`:

```python
# Liveness Sensitivity
MOUTH_AR_THRESH = 0.60      # Higher = Mouth must open wider
YAW_THRESH_LEFT = 15        # Angle required to pass "Look Left"
SIMILARITY_THRESHOLD = 0.45 # Lower = Easier to match identity (Less secure)

```

## ⚠️ Troubleshooting

**1. "No Face Detected" during Liveness**
Ensure you have good lighting. MediaPipe Face Mesh requires the face to be clearly visible. Backlighting (a window behind you) often causes detection failures.

**2. Protobuf Errors**
If you see errors related to `Symbol not found` or `Protocol Buffers`, uninstall `protobuf` and reinstall the specific version from requirements:

```bash
pip uninstall protobuf
pip install protobuf==3.20.3

```