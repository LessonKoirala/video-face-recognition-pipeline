import cv2
import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer

# Paths
emb_cache_dir = "frames/_emb_cache"
video_path = "videos/new_video.mp4"
output_path = "videos/output_recognized.mp4"

# Load embeddings
def load_embeddings(emb_cache_dir):
    X, y = [], []
    for fname in os.listdir(emb_cache_dir):
        if not fname.endswith(".npy"):
            continue
        label = os.path.splitext(fname)[0]  # "person1", "person2"
        embeddings = np.load(os.path.join(emb_cache_dir, fname))
        for emb in embeddings:
            X.append(emb)
            y.append(label)
    return np.array(X), np.array(y)

print("[INFO] Loading embeddings...")
X, y = load_embeddings(emb_cache_dir)

# Normalize
norm = Normalizer(norm="l2")
X = norm.transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# Train classifier
print("[INFO] Training classifier...")
model = SVC(kernel="linear", probability=True)
model.fit(X, y_enc)

# Save model for reuse
with open("face_classifier.pkl", "wb") as f:
    pickle.dump((model, label_encoder, norm), f)

# Load face detector (Haar Cascade for simplicity)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Import your embedding extractor
from embedded import extract_embedding

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Could not open video: {video_path}")
    exit()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("[INFO] Processing video...")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processing frame {frame_count}...")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y0, w, h) in faces:
        face = frame[y0:y0+h, x:x+w]

        try:
            # Convert face to embedding
            emb = extract_embedding(face)
            emb = norm.transform([emb])

            # Get prediction and confidence
            probs = model.predict_proba(emb)[0]
            class_index = np.argmax(probs)
            confidence = probs[class_index]
            person_name = label_encoder.inverse_transform([class_index])[0]

            # Set color and label based on confidence
            if confidence > 0.6:
                color = (0, 255, 0)  # green
                label = f"{person_name} ({confidence:.2f})"
            else:
                color = (0, 0, 255)  # red
                label = f"{person_name}? ({confidence:.2f})"

        except Exception as e:
            print(f"[ERROR] {e}")
            color = (0, 0, 255)
            label = "Error"

        # Draw bounding box
        cv2.rectangle(frame, (x, y0), (x+w, y0+h), color, 2)
        cv2.putText(frame, label, (x, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out.write(frame)

cap.release()
out.release()
print(f"[INFO] Saved output video to {output_path}")