---

# Face Recognition Project

This project implements a **video-based face recognition pipeline** using Python.
It allows you to:

* Extract frames from videos
* Detect and classify faces
* Generate embeddings for similarity comparison
* Recognize people in new videos

---

## 📂 Project Structure

```
face-recong/
│── detect_face.py          # Face detection & recognition from video
│── embedded.py             # Embedding generator & similarity metrics
│── extract_frame.py        # Extracts frames from videos
│── face_classifier.pkl     # Pre-trained face classifier (pickle file)
│── videos/                 # Input/output videos
│   ├── new_video.mp4
│   ├── output_recognized.mp4
│   ├── person1/
│   └── person2/
│── frames/                 # Extracted frames & embeddings
│   ├── person1/
│   └── person2/

---

## ⚙️ Requirements

* Python 3.8+
* Install dependencies:

```bash
pip install -r requirements.txt
```

If no `requirements.txt` is included, you’ll likely need:

```bash
pip install opencv-python numpy scikit-learn matplotlib
```

---

## 🚀 Usage

### 1. Extract frames from videos

Run the script to split a video into frames per person:

```bash
python extract_frame.py
```

* Saves frames into `frames/person1/` and `frames/person2/`.

---

### 2. Generate face embeddings

Compute embeddings from the frames:

```bash
python embedded.py
```

* Produces `.npy` embedding files inside `frames/_emb_cache/`.
* Prints centroid and separation metrics to check how distinct the faces are.

---

### 3. Detect and recognize faces in a video

Run the recognition pipeline:

```bash
python detect_face.py
```

* Takes `videos/new_video.mp4` as input.
* Outputs `videos/output_recognized.mp4` with recognized faces highlighted.

---

## 🎯 Example Workflow

1. Place your training videos/images into `videos/person1/` and `videos/person2/`.

2. Extract frames:

   ```bash
   python extract_frame.py
   ```

3. Train embeddings and check metrics:

   ```bash
   python embedded.py
   ```

4. Run recognition on a new video:

   ```bash
   python detect_face.py
   ```

   The result will be saved as `videos/output_recognized.mp4`.

---

## 📌 Notes

* `face_classifier.pkl` stores the trained face classifier (SVM/KNN).
* Embedding vectors are 512-dimensional and compared using cosine & Euclidean distances.
* You can add more people by creating new folders (`videos/person3/`, `frames/person3/`) and re-running the embedding pipeline.
