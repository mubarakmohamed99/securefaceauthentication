# extract_embeddings.py

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis # type: ignore
import os
import pickle
import json

FACE_DB_PATH = "faces_db"
SAVE_DIR = "face_embeddings"
METADATA_FILE = os.path.join(SAVE_DIR, "metadata.json")

def load_existing_metadata():
    """Load existing metadata to check already processed persons."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f).get("names", [])
    return []

def load_face_database():
    """Load images for only new persons, compute average embeddings, and store them."""
    if not os.path.exists(FACE_DB_PATH):
        print(f"Error: Face database folder '{FACE_DB_PATH}' not found.")
        return {}

    # Load existing metadata to avoid reprocessing
    existing_names = set(load_existing_metadata())

    face_analyzer = FaceAnalysis(
        name='buffalo_s', 
        providers=['CPUExecutionProvider'], 
        allowed_modules=['detection', 'recognition']
    )
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    known_faces = {}

    person_folders = [p for p in os.listdir(FACE_DB_PATH) if os.path.isdir(os.path.join(FACE_DB_PATH, p)) and p not in existing_names]
    total_persons = len(person_folders)

    for idx, person_name in enumerate(person_folders, start=1):
        person_path = os.path.join(FACE_DB_PATH, person_name)
        embeddings = []

        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read {image_path}")
                continue

            # Convert BGR to RGB (InsightFace expects RGB input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = face_analyzer.get(image)

            if len(faces) == 1:  # Use only images with exactly one face
                embeddings.append(faces[0].embedding)
            else:
                print(f"Skipping {image_path}: Found {len(faces)} faces")

        if embeddings:
            # Compute the mean embedding
            avg_embedding = np.mean(embeddings, axis=0)
            known_faces[person_name] = avg_embedding
            print(f"({idx}/{total_persons}) Stored embedding for {person_name} ({len(embeddings)} images)")
        else:
            print(f"({idx}/{total_persons}) No valid images found for {person_name}")

    return known_faces

def save_embeddings(known_faces):
    """Save computed embeddings and update metadata."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Load existing metadata
    existing_names = set(load_existing_metadata())

    # Update metadata with new names
    updated_names = list(existing_names.union(known_faces.keys()))
    with open(METADATA_FILE, "w") as f:
        json.dump({"names": updated_names}, f)

    for name, embedding in known_faces.items():
        embedding_path = os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}.pkl")
        with open(embedding_path, "wb") as f:
            pickle.dump(embedding, f)

    print(f"Saved {len(known_faces)} new face embeddings to {SAVE_DIR}")

if __name__ == "__main__":
    new_faces = load_face_database()
    if new_faces:
        save_embeddings(new_faces)
    else:
        print("No new persons detected. Skipping extraction process.")
