import cv2
import numpy as np
import time
import os
import pickle
import mediapipe as mp
from scipy.spatial import distance as dist
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION
# ==========================================
MOUTH_AR_THRESH = 0.60      # Threshold to consider mouth open
YAW_THRESH_LEFT = 15        # Degrees to look left
YAW_THRESH_RIGHT = -15      # Degrees to look right
PITCH_THRESH_UP = 10        # Degrees to look up
PITCH_THRESH_DOWN = -10     # Degrees to look down

# Identity Thresholds
SIMILARITY_THRESHOLD = 0.45 # Slightly lower than 0.5 to account for webcam variations

# ==========================================
# LIVENESS DETECTOR (Unchanged)
# ==========================================
class LivenessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        # Challenges: Left -> Right -> Open Mouth -> Nod Up
        self.challenges = ["look_left", "look_right", "open_mouth", "nod_up"]
        self.current_step = 0
        self.success = False
        self.instruction_text = "Center your face"
        self.status_color = (255, 255, 255)

    def get_head_pose(self, shape, img_h, img_w):
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        image_points = np.array([
            (shape[1].x * img_w, shape[1].y * img_h),
            (shape[152].x * img_w, shape[152].y * img_h),
            (shape[33].x * img_w, shape[33].y * img_h),
            (shape[263].x * img_w, shape[263].y * img_h),
            (shape[61].x * img_w, shape[61].y * img_h),
            (shape[291].x * img_w, shape[291].y * img_h)
        ], dtype="double")

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        
        success, vector_rotation, vector_translation = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
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
        pitch, yaw, roll = self.get_head_pose(shape, img_h, img_w)
        mar = self.get_mouth_ratio(shape, img_h, img_w)
        
        current_challenge = self.challenges[self.current_step]
        challenge_met = False

        if current_challenge == "look_left":
            self.instruction_text = "Turn Head LEFT"
            if yaw > YAW_THRESH_LEFT: challenge_met = True
        elif current_challenge == "look_right":
            self.instruction_text = "Turn Head RIGHT"
            if yaw < YAW_THRESH_RIGHT: challenge_met = True
        elif current_challenge == "open_mouth":
            self.instruction_text = "Open Your MOUTH"
            if mar > MOUTH_AR_THRESH: challenge_met = True
        elif current_challenge == "nod_up":
            self.instruction_text = "Nod UP"
            if pitch > PITCH_THRESH_UP: challenge_met = True

        if challenge_met:
            self.status_color = (0, 255, 0)
            self.current_step += 1
            if self.current_step >= len(self.challenges):
                self.success = True
                self.instruction_text = "Liveness Verified!"
        else:
            self.status_color = (255, 255, 255)

        return frame, self.success

# ==========================================
# MAIN SYSTEM (Fixed Verification Logic)
# ==========================================
class IdentitySystem:
    def __init__(self):
        self.liveness = LivenessDetector()
        
        # Use exact config from your working script
        self.app = FaceAnalysis(
            name='buffalo_s', 
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.known_faces = {}
        self.data_dir = "face_embeddings"
        self.load_embeddings()

    def load_embeddings(self):
        if not os.path.exists(self.data_dir):
            print("No embedding directory found.")
            return
        
        count = 0
        for f in os.listdir(self.data_dir):
            if f.endswith('.pkl'):
                name = os.path.splitext(f)[0]
                # Sanitize name to match how it might be saved
                name = name.replace("_", " ") 
                with open(os.path.join(self.data_dir, f), "rb") as file:
                    self.known_faces[name] = pickle.load(file)
                    count += 1
        print(f"Loaded {count} identities.")

    def recognize(self, frame):
        # Scale down to match the working script's logic (improves speed and detection)
        scale_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        
        faces = self.app.get(small_frame)
        if len(faces) == 0:
            return None, 0.0
        
        # Get largest face
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        
        best_match = None
        best_score = 0.0

        for name, ref_emb in self.known_faces.items():
            # Calculate Cosine Similarity
            sim = np.dot(face.embedding, ref_emb) / (np.linalg.norm(face.embedding) * np.linalg.norm(ref_emb))
            if sim > best_score:
                best_score = sim
                best_match = name
        
        if best_score >= SIMILARITY_THRESHOLD:
            return best_match, best_score
        return None, best_score

    def draw_ui(self, frame, text, color, verified=False, user_name="", score=0.0):
        h, w, _ = frame.shape
        
        # 1. Create Dark Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.7
        
        # 2. Create Circular Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = int(min(h, w) * 0.35)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Combine
        mask_inv = cv2.bitwise_not(mask)
        temp = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        final_frame = cv2.bitwise_and(temp, temp, mask=mask_inv)
        face_roi = cv2.bitwise_and(frame, frame, mask=mask)
        final_frame = cv2.add(final_frame, face_roi)

        # 3. Draw Circle Border
        cv2.circle(final_frame, center, radius, color, 4)
        
        # 4. Draw Instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if verified:
            # --- Success Screen ---
            cv2.putText(final_frame, "VERIFICATION COMPLETE", (w//2 - 180, 80), font, 0.8, (0, 255, 0), 2)
            
            # 1. Display Confidence (Small, above name)
            conf_text = f"Confidence: {score*100:.1f}%"
            conf_size = cv2.getTextSize(conf_text, font, 0.6, 1)[0]
            conf_x = (w - conf_size[0]) // 2
            cv2.putText(final_frame, conf_text, (conf_x, h - 80), font, 0.6, (200, 255, 200), 1)

            # 2. Display Name (Large, centered at bottom)
            text_welcome = f"Welcome, {user_name.title()}"
            text_size = cv2.getTextSize(text_welcome, font, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(final_frame, text_welcome, (text_x, h - 40), font, 1.0, (0, 255, 0), 2)
            
        else:
            # --- Challenge Screen ---
            cv2.putText(final_frame, text, (w//2 - 120, 80), font, 1.0, color, 2)
            
            # Progress Dots
            total_steps = len(self.liveness.challenges)
            curr = self.liveness.current_step
            start_x = w//2 - (total_steps * 20)
            dots_y = h - 40 
            for i in range(total_steps):
                c = (0, 255, 0) if i < curr else (100, 100, 100)
                cv2.circle(final_frame, (start_x + i*40, dots_y), 10, c, -1)

        return final_frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        verified_identity = None
        verification_confidence = 0.0
        
        # Timer state
        verification_start_time = None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            # 1. Process Liveness
            frame, is_live = self.liveness.process(frame)
            
            display = frame
            
            # 2. If Liveness Passed -> Start Identity Verification Loop
            if is_live:
                # If we haven't found a match yet, keep trying
                if verified_identity is None:
                    if verification_start_time is None:
                        verification_start_time = time.time()
                    
                    # Try to recognize
                    name, score = self.recognize(frame)
                    
                    if name:
                        verified_identity = name
                        verification_confidence = score
                    else:
                        # Feedback while verifying
                        display = self.draw_ui(frame, "Verifying Identity...", (0, 255, 255))
                
                # If we found a match, show success
                if verified_identity:
                    display = self.draw_ui(frame, "Done", (0, 255, 0), True, verified_identity, verification_confidence)
                
            else:
                # Still doing liveness challenges
                display = self.draw_ui(frame, self.liveness.instruction_text, self.liveness.status_color, False)

            cv2.imshow("Secure Verification", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = IdentitySystem()
    system.run()