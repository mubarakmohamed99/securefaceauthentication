import streamlit as st
import cv2
import numpy as np
import time
import os
import pickle
import mediapipe as mp
from scipy.spatial import distance as dist
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
MOUTH_AR_THRESH = 0.60
YAW_THRESH_LEFT = 15
YAW_THRESH_RIGHT = -15
PITCH_THRESH_UP = 10
PITCH_THRESH_DOWN = -10
SIMILARITY_THRESHOLD = 0.45
FACE_DB_PATH = "faces_db"
SAVE_DIR = "face_embeddings"

# ==========================================
# LIVENESS DETECTOR CLASS
# ==========================================
class LivenessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        self.challenges = ["look_left", "look_right", "open_mouth", "nod_up"]
        self.current_step = 0
        self.success = False
        self.instruction_text = "Center your face"
        self.status_color = (255, 255, 255)

    def get_head_pose(self, shape, img_h, img_w):
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])
        image_points = np.array([
            (shape[1].x * img_w, shape[1].y * img_h), (shape[152].x * img_w, shape[152].y * img_h),
            (shape[33].x * img_w, shape[33].y * img_h), (shape[263].x * img_w, shape[263].y * img_h),
            (shape[61].x * img_w, shape[61].y * img_h), (shape[291].x * img_w, shape[291].y * img_h)
        ], dtype="double")
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        _, vector_rotation, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        rmat, _ = cv2.Rodrigues(vector_rotation)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0] * 360, angles[1] * 360, angles[2] * 360

    def get_mouth_ratio(self, shape, img_h, img_w):
        top = (shape[13].x * img_w, shape[13].y * img_h)
        bottom = (shape[14].x * img_w, shape[14].y * img_h)
        left = (shape[61].x * img_w, shape[61].y * img_h)
        right = (shape[291].x * img_w, shape[291].y * img_h)
        return dist.euclidean(top, bottom) / dist.euclidean(left, right)

    def process(self, frame):
        if self.success: return frame, True
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            self.instruction_text = "No Face Detected"
            self.status_color = (0, 0, 255)
            return frame, False
        shape = results.multi_face_landmarks[0].landmark
        pitch, yaw, _ = self.get_head_pose(shape, img_h, img_w)
        mar = self.get_mouth_ratio(shape, img_h, img_w)
        current_challenge = self.challenges[self.current_step]
        challenge_met = False
        if current_challenge == "look_left" and yaw > YAW_THRESH_LEFT: challenge_met = True
        elif current_challenge == "look_right" and yaw < YAW_THRESH_RIGHT: challenge_met = True
        elif current_challenge == "open_mouth" and mar > MOUTH_AR_THRESH: challenge_met = True
        elif current_challenge == "nod_up" and pitch > PITCH_THRESH_UP: challenge_met = True
        
        self.instruction_text = current_challenge.replace("_", " ").upper()
        if challenge_met:
            self.status_color = (0, 255, 0)
            self.current_step += 1
            if self.current_step >= len(self.challenges):
                self.success = True
                self.instruction_text = "Verified!"
        else: self.status_color = (255, 255, 255)
        return frame, self.success

# ==========================================
# IDENTITY SYSTEM CLASS
# ==========================================
class IdentitySystem:
    def __init__(self):
        self.liveness = LivenessDetector()
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces = {}
        self.data_dir = SAVE_DIR
        self.load_embeddings()

    def load_embeddings(self):
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        for f in os.listdir(self.data_dir):
            if f.endswith('.pkl'):
                name = os.path.splitext(f)[0].replace("_", " ")
                with open(os.path.join(self.data_dir, f), "rb") as file:
                    self.known_faces[name] = pickle.load(file)

    def recognize(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        faces = self.app.get(small_frame)
        if not faces: return None, 0.0
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        best_match, best_score = None, 0.0
        for name, ref_emb in self.known_faces.items():
            sim = np.dot(face.embedding, ref_emb) / (np.linalg.norm(face.embedding) * np.linalg.norm(ref_emb))
            if sim > best_score:
                best_score = sim
                best_match = name
        return (best_match, best_score) if best_score >= SIMILARITY_THRESHOLD else (None, best_score)

    def draw_ui(self, frame, text, color, verified=False, user_name="", score=0.0):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        mask = np.zeros((h, w), dtype=np.uint8)
        center, radius = (w // 2, h // 2), int(min(h, w) * 0.35)
        cv2.circle(mask, center, radius, 255, -1)
        final_frame = cv2.bitwise_and(cv2.addWeighted(overlay, 0.7, frame, 0.3, 0), cv2.addWeighted(overlay, 0.7, frame, 0.3, 0), mask=cv2.bitwise_not(mask))
        final_frame = cv2.add(final_frame, cv2.bitwise_and(frame, frame, mask=mask))
        cv2.circle(final_frame, center, radius, color, 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if verified:
            cv2.putText(final_frame, f"WELCOME, {user_name.upper()}", (w//2-150, h-40), font, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(final_frame, text, (w//2-100, 80), font, 0.8, color, 2)
        return final_frame

# ==========================================
# STREAMLIT UI & NAVIGATION
# ==========================================
def main():
    st.set_page_config(page_title="SecureGate AI", layout="centered")
    
    if 'system' not in st.session_state:
        with st.spinner("Loading AI Models..."):
            st.session_state.system = IdentitySystem()
        st.session_state.page = "landing"

    # --- LANDING PAGE ---
    if st.session_state.page == "landing":
        st.title("🛡️ SecureGate AI")
        st.write("Biometric Liveness & Identity Verification")
        if st.button("Login", use_container_width=True):
            st.session_state.system.liveness = LivenessDetector() # Reset liveness
            st.session_state.page = "login"; st.rerun()
        if st.button("Register", use_container_width=True):
            st.session_state.page = "register"; st.rerun()

    # --- REGISTER PAGE ---
    elif st.session_state.page == "register":
        st.title("📝 User Enrollment")
        POSES = ["Look Straight", "Tilt Right", "Tilt Left", "Smile", "Look Up"]
        if 'reg_step' not in st.session_state: st.session_state.reg_step = 0
        if 'reg_embs' not in st.session_state: st.session_state.reg_embs = []
        
        name = st.text_input("Full Name")
        if name:
            if st.session_state.reg_step < len(POSES):
                st.subheader(f"Pose: {POSES[st.session_state.reg_step]}")
                img = st.camera_input("Capture", key=f"p{st.session_state.reg_step}")
                if img:
                    # Save physical image
                    p_path = os.path.join(FACE_DB_PATH, name.replace(" ","_"))
                    if not os.path.exists(p_path): os.makedirs(p_path)
                    with open(os.path.join(p_path, f"{st.session_state.reg_step}.jpg"), "wb") as f: f.write(img.getbuffer())
                    # Extract embedding
                    f_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
                    frame = cv2.imdecode(f_bytes, 1)
                    faces = st.session_state.system.app.get(frame)
                    if faces:
                        st.session_state.reg_embs.append(faces[0].embedding)
                        st.session_state.reg_step += 1; st.rerun()
            else:
                if st.button("Finalize Registration"):
                    avg_emb = np.mean(st.session_state.reg_embs, axis=0)
                    with open(os.path.join(SAVE_DIR, f"{name.replace(' ','_')}.pkl"), "wb") as f: pickle.dump(avg_emb, f)
                    st.session_state.system.known_faces[name] = avg_emb
                    st.session_state.reg_step = 0; st.session_state.reg_embs = []; st.session_state.page = "landing"; st.rerun()
        if st.button("Back"): st.session_state.page = "landing"; st.rerun()

    # --- LOGIN PAGE ---
    elif st.session_state.page == "login":
        st.title("🔒 Biometric Login")
        frame_placeholder = st.empty()
        if st.button("Stop"): st.session_state.page = "landing"; st.rerun()
        
        cap = cv2.VideoCapture(0)
        verified_user = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame, is_live = st.session_state.system.liveness.process(frame)
            
            if is_live:
                name, score = st.session_state.system.recognize(frame)
                if name:
                    verified_user = name
                    display = st.session_state.system.draw_ui(frame, "Access Granted", (0,255,0), True, name, score)
                else: display = st.session_state.system.draw_ui(frame, "Verifying...", (0,255,255))
            else:
                display = st.session_state.system.draw_ui(frame, st.session_state.system.liveness.instruction_text, st.session_state.system.liveness.status_color)

            frame_placeholder.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB")
            if verified_user:
                time.sleep(2); break
        cap.release()
        st.session_state.page = "landing"; st.rerun()

if __name__ == "__main__":
    main()
