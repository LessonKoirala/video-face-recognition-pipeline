# embed_all_in_one.py  (with centroid metrics)
import os, glob, argparse, json
from typing import List, Tuple, Dict, Optional
import numpy as np, cv2, torch, torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

FRAMES_DIR = "frames"
P1_DIR = os.path.join(FRAMES_DIR, "person1")
P2_DIR = os.path.join(FRAMES_DIR, "person2")
IMG_SIZE = 224
BATCH_SIZE = 32
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
CACHE_DIR = os.path.join(FRAMES_DIR, "_emb_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize model and utilities
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class ResNet18Feat(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = self.backbone(x)
            return y.view(y.size(0), -1)

_model = ResNet18Feat().to(_device).eval()

_preproc = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

def detect_face_bbox(img_bgr: np.ndarray) -> Tuple[int,int,int,int]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _haar.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
    if len(faces) > 0:
        x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
        return int(x),int(y),int(w),int(h)
    
    # If no face detected, use the entire image
    h_img, w_img = img_bgr.shape[:2]
    # Use a smaller square in the center for already cropped faces
    side = min(h_img, w_img) * 0.8  # Use 80% of the smaller dimension
    cx, cy = w_img//2, h_img//2
    x = max(0, cx - side//2)
    y = max(0, cy - side//2)
    return int(x), int(y), int(side), int(side)

def extract_embedding(face_img):
    """
    Input: face image (numpy array, BGR)
    Output: embedding (1D numpy array)
    """
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Detect face in the image (even though it's already a face crop, we'll use the center crop)
    x, y, w, h = detect_face_bbox(face_img)
    
    # Crop and resize the face
    face_crop = face_rgb[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    
    # Preprocess for the model
    tensor = _preproc(face_resized).unsqueeze(0).to(_device)
    
    # Extract embedding
    with torch.no_grad():
        embedding = _model(tensor)
    
    return embedding.cpu().numpy().flatten()

def load_and_prepare(paths: List[str]) -> torch.Tensor:
    ts=[]
    for p in paths:
        im = cv2.imread(p)
        if im is None: continue
        x,y,w,h = detect_face_bbox(im)
        crop = im[y:y+h, x:x+w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ts.append(_preproc(crop_rgb))
    return torch.stack(ts,0) if ts else torch.empty(0,3,IMG_SIZE,IMG_SIZE)

def list_images(d: str) -> List[str]:
    out=[]
    for e in ("*.jpg","*.jpeg","*.png","*.bmp"):
        out.extend(glob.glob(os.path.join(d,e)))
    return sorted(out)

def extract_embeddings_for_dir(dir_path: str, cache_key: str, rebuild: bool) -> np.ndarray:
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.isfile(cache_file) and not rebuild:
        return np.load(cache_file)
    imgs = list_images(dir_path)
    if not imgs:
        print(f"[WARN] No images found in {dir_path}")
        emb = np.empty((0,512), np.float32); np.save(cache_file, emb); return emb
    feats=[]
    for i in range(0,len(imgs),BATCH_SIZE):
        batch = load_and_prepare(imgs[i:i+BATCH_SIZE])
        if batch.numel()==0: continue
        vecs = _model(batch.to(_device, non_blocking=True))
        feats.append(vecs.cpu().numpy().astype(np.float32))
    emb = np.vstack(feats) if feats else np.empty((0,512), np.float32)
    np.save(cache_file, emb)
    return emb

def build_embeddings(rebuild=False) -> Dict[str,np.ndarray]:
    return {
        "person1": extract_embeddings_for_dir(P1_DIR, "person1", rebuild),
        "person2": extract_embeddings_for_dir(P2_DIR, "person2", rebuild),
    }

def pca(X: np.ndarray, k: int) -> np.ndarray:
    if X.ndim!=2 or X.shape[0]==0: return np.empty((0,k), np.float32)
    if X.shape[1] < k: raise ValueError(f"PCA k={k} > D={X.shape[1]}")
    Xc = X - X.mean(axis=0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:k].T).astype(np.float32)

# ---------- NEW: centroid + distance metrics ----------
def l2_normalize(X: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)

def centroid(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0, dtype=np.float64)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 1 - cosine similarity
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a_n, b_n))

def mean_dist_to_centroid(X: np.ndarray, c: np.ndarray) -> Tuple[float,float]:
    # Euclidean and cosine mean distance to centroid
    if X.shape[0]==0: return float("nan"), float("nan")
    eu = np.linalg.norm(X - c[None,:], axis=1).mean()
    Xn = l2_normalize(X.astype(np.float64))
    cn = c / (np.linalg.norm(c) + 1e-12)
    cos = (1.0 - (Xn @ cn)).mean()
    return float(eu), float(cos)

def separation_report(p1: np.ndarray, p2: np.ndarray) -> Dict[str,float]:
    # raw centroids
    c1_raw, c2_raw = centroid(p1), centroid(p2)
    between_l2  = float(np.linalg.norm(c1_raw - c2_raw))
    between_cos = cosine_distance(c1_raw, c2_raw)

    # within spreads
    w1_l2, w1_cos = mean_dist_to_centroid(p1, c1_raw)
    w2_l2, w2_cos = mean_dist_to_centroid(p2, c2_raw)
    within_l2 = (w1_l2 + w2_l2) / 2.0
    within_cos = (w1_cos + w2_cos) / 2.0

    return {
        "between_l2": between_l2,
        "between_cosine": between_cos,
        "within_l2_mean": within_l2,
        "within_cosine_mean": within_cos,
        "person1_within_l2": w1_l2,
        "person2_within_l2": w2_l2,
        "person1_within_cosine": w1_cos,
        "person2_within_cosine": w2_cos,
        "separation_ratio_l2": between_l2 / max(within_l2, 1e-12),
        "separation_ratio_cosine": between_cos / max(within_cos, 1e-12),
    }

# ---------- plotting ----------
def plot_2d(emb, title, ax=None):
    if emb.size == 0:
        if ax is not None: ax.set_title(title + " (no data)"); return
    if ax is None: fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(emb[:,0], emb[:,1], ".", alpha=0.8, markersize=4)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

def plot_3d(emb, title, ax=None):
    if emb.size == 0:
        if ax is not None: ax.set_title(title + " (no data)"); return
    if ax is None:
        fig = plt.figure(figsize=(6,5)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(emb[:,0], emb[:,1], emb[:,2], ".", alpha=0.8, markersize=4)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

def make_plots(p1, p2, show=True, overlay_centroids=True):
    out_dir = os.path.join(FRAMES_DIR, "_plots"); os.makedirs(out_dir, exist_ok=True)
    p1_2d = pca(p1,2) if p1.shape[1]>=2 else np.empty((0,2), np.float32)
    p2_2d = pca(p2,2) if p2.shape[1]>=2 else np.empty((0,2), np.float32)
    p1_3d = pca(p1,3) if p1.shape[1]>=3 else np.empty((0,3), np.float32)
    p2_3d = pca(p2,3) if p2.shape[1]>=3 else np.empty((0,3), np.float32)
    both = np.vstack([p1,p2]) if (p1.size and p2.size) else (p1 if p1.size else p2)
    both_2d = pca(both,2) if both.shape[1]>=2 else np.empty((0,2), np.float32)
    both_3d = pca(both,3) if both.shape[1]>=3 else np.empty((0,3), np.float32)

    if p1.size and p2.size and both_2d.size:
        p1_both_2d = both_2d[:len(p1)]; p2_both_2d = both_2d[len(p1):]
    else:
        p1_both_2d, p2_both_2d = p1_2d, p2_2d
    if p1.size and p2.size and both_3d.size:
        p1_both_3d = both_3d[:len(p1)]; p2_both_3d = both_3d[len(p1):]
    else:
        p1_both_3d, p2_both_3d = p1_3d, p2_3d

    fig2d, axes2d = plt.subplots(1,3, figsize=(16,5))
    plot_2d(p1_2d, "Person 1 — 2D (PCA on Person 1)", ax=axes2d[0])
    plot_2d(p2_2d, "Person 2 — 2D (PCA on Person 2)", ax=axes2d[1])
    ax_mix = axes2d[2]
    if p1_both_2d.size: ax_mix.plot(p1_both_2d[:,0], p1_both_2d[:,1], ".", label="Person 1", alpha=0.7)
    if p2_both_2d.size: ax_mix.plot(p2_both_2d[:,0], p2_both_2d[:,1], "x", label="Person 2", alpha=0.7)
    ax_mix.set_title("Mixed — 2D (PCA on Both)"); ax_mix.set_xlabel("PC1"); ax_mix.set_ylabel("PC2"); ax_mix.legend()
    fig2d.tight_layout(); fig2d.savefig(os.path.join(out_dir,"pca_2d.png"), dpi=150)

    # Mixed 3D
    if both_3d.size:
        fig = plt.figure(figsize=(7,6)); ax = fig.add_subplot(111, projection="3d")
        if p1_both_3d.size: ax.plot(p1_both_3d[:,0], p1_both_3d[:,1], p1_both_3d[:,2], ".", label="Person 1", alpha=0.7)
        if p2_both_3d.size: ax.plot(p2_both_3d[:,0], p2_both_3d[:,1], p2_both_3d[:,2], "x", label="Person 2", alpha=0.7)
        ax.set_title("Mixed — 3D (PCA on Both)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.legend()
        fig.savefig(os.path.join(out_dir,"pca_3d.png"), dpi=150)

    # Individual 3D
    if p1_3d.size:
        fig = plt.figure(figsize=(6,5)); ax = fig.add_subplot(111, projection="3d")
        plot_3d(p1_3d, "Person 1 — 3D (PCA on Person 1)", ax=ax)
        fig.savefig(os.path.join(out_dir,"person1_pca_3d.png"), dpi=150)
    if p2_3d.size:
        fig = plt.figure(figsize=(6,5)); ax = fig.add_subplot(111, projection="3d")
        plot_3d(p2_3d, "Person 2 — 3D (PCA on Person 2)", ax=ax)
        fig.savefig(os.path.join(out_dir,"person2_pca_3d.png"), dpi=150)

    if show: plt.show()
    print(f"[INFO] Saved plots to {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Embeddings, PCA plots, and centroid metrics")
    ap.add_argument("--rebuild", action="store_true", help="Force re-extraction")
    ap.add_argument("--no-show", action="store_true", help="Do not display interactive plots")
    args = ap.parse_args()

    for d in (P1_DIR, P2_DIR):
        if not os.path.isdir(d): raise SystemExit(f"Missing folder: {d}")

    embs = build_embeddings(rebuild=args.rebuild)
    p1, p2 = embs["person1"], embs["person2"]
    print(f"person1 embeddings: {p1.shape}, dtype={p1.dtype}")
    print(f"person2 embeddings: {p2.shape}, dtype={p2.dtype}")

    # ---- NEW: print and save centroid/separation metrics ----
    metrics = separation_report(p1, p2)
    print("\n=== Centroid / Separation Metrics (512D) ===")
    print(f"Between-centroid Euclidean: {metrics['between_l2']:.4f}")
    print(f"Between-centroid Cosine   : {metrics['between_cosine']:.6f}  (0 means identical, 2 means opposite)")
    print(f"Within (mean Euclid)      : {metrics['within_l2_mean']:.4f}  "
          f"[p1: {metrics['person1_within_l2']:.4f}, p2: {metrics['person2_within_l2']:.4f}]")
    print(f"Within (mean Cosine)      : {metrics['within_cosine_mean']:.6f}  "
          f"[p1: {metrics['person1_within_cosine']:.6f}, p2: {metrics['person2_within_cosine']:.6f}]")
    print(f"Separation ratio (L2)     : {metrics['separation_ratio_l2']:.3f}  "
          f"(>1 means between > within)")
    print(f"Separation ratio (Cosine) : {metrics['separation_ratio_cosine']:.3f}")

    out_dir = os.path.join(FRAMES_DIR, "_plots"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v}\n")

    # ---- plots ----
    if p1.shape[1] < 2 or p2.shape[1] < 2:
        raise SystemExit("Embeddings must have >=2 dims for plots.")
    make_plots(p1, p2, show=not args.no_show)

if __name__ == "__main__":
    main()