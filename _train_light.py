"""Retrain lightweight model — rich features + HistGradientBoosting."""
import os, pickle
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET = os.path.join(os.path.dirname(__file__), '..', 'ai-models', 'dataset')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_MAP = {
    '1. Tea algal leaf spot': 'Tea Algal Leaf Spot',
    '2. Brown Blight':        'Brown Blight',
    '3. Gray Blight':         'Gray Blight',
    '4. Helopeltis':          'Helopeltis',
    '5. Red spider':          'Red Spider',
    '6. Green mirid bug':     'Green Mirid Bug',
    '7. Healthy leaf':        'Healthy',
}

def rgb_to_hsv(arr):
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 1e-6, (maxc - minc) / maxc, 0.0)
    diff = np.where(maxc - minc > 1e-6, maxc - minc, 1.0)
    rc = (maxc - r) / diff
    gc = (maxc - g) / diff
    bc = (maxc - b) / diff
    h = np.where(r == maxc, bc - gc,
        np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=2)

def extract(path):
    img = Image.open(path).convert('RGB').resize((128, 128), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    hsv  = rgb_to_hsv(arr)
    ycbcr = np.array(img.convert('YCbCr'), dtype=np.float32) / 255.0
    feats = []

    # 1. Histograms: RGB + HSV + YCbCr (32 bins each)
    for space in [arr, hsv, ycbcr]:
        for c in range(3):
            h, _ = np.histogram(space[:,:,c], bins=32, range=(0, 1))
            feats.extend(h / (h.sum() + 1e-8))

    # 2. Per-channel stats (mean, std, q10, q25, q75, q90)
    for c in range(3):
        ch = arr[:,:,c]
        feats += [ch.mean(), ch.std(),
                  float(np.percentile(ch, 10)), float(np.percentile(ch, 25)),
                  float(np.percentile(ch, 75)), float(np.percentile(ch, 90))]

    # 3. 4×4 spatial grid — mean + std per RGB channel
    S = 32
    for qi in range(4):
        for qj in range(4):
            q = arr[qi*S:(qi+1)*S, qj*S:(qj+1)*S]
            for c in range(3):
                feats += [q[:,:,c].mean(), q[:,:,c].std()]

    # 4. Gradient magnitude at 3 scales
    for scale in [128, 64, 32]:
        img_s = img.resize((scale, scale), Image.LANCZOS)
        a = np.array(img_s, dtype=np.float32) / 255.0
        gx = np.abs(a[:,1:,:] - a[:,:-1,:]).mean()
        gy = np.abs(a[1:,:,:] - a[:-1,:,:]).mean()
        feats += [gx, gy, (gx + gy) / 2]

    # 5. LBP-like texture on grayscale
    gray = arr.mean(axis=2)
    pad = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, 1:-1], gray[:-2, 2:],  gray[1:-1, 2:], gray[2:, 2:],
        gray[2:,  1:-1], gray[2:,  :-2], gray[1:-1,:-2], gray[:-2, :-2],
    ]
    codes = sum((n >= pad).astype(np.uint8) << i for i, n in enumerate(neighbors))
    lbp_h, _ = np.histogram(codes, bins=32, range=(0, 255))
    feats.extend(lbp_h / (lbp_h.sum() + 1e-8))

    # 6. Hue quadrant ratios
    h_ch = hsv[:,:,0]
    feats += [
        ((h_ch < 0.08) | (h_ch > 0.92)).mean(),
        ((h_ch >= 0.08) & (h_ch < 0.22)).mean(),
        ((h_ch >= 0.22) & (h_ch < 0.45)).mean(),
        ((h_ch >= 0.45) & (h_ch < 0.65)).mean(),
        ((h_ch >= 0.65) & (h_ch < 0.78)).mean(),
        ((h_ch >= 0.78) & (h_ch < 0.92)).mean(),
    ]

    # 7. Saturation and brightness stats
    feats += [hsv[:,:,1].mean(), hsv[:,:,1].std(),
              hsv[:,:,2].mean(), hsv[:,:,2].std()]

    # 8. Dark/bright/saturated pixel ratios
    feats += [
        (arr.mean(axis=2) < 0.25).mean(),   # dark pixels
        (arr.mean(axis=2) > 0.75).mean(),   # bright pixels
        (hsv[:,:,1] > 0.5).mean(),          # saturated pixels
    ]

    return np.array(feats, dtype=np.float32)

# Load dataset
X, y = [], []
folders = sorted(d for d in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET, d)))
for folder in folders:
    cls = CLASS_MAP.get(folder, folder)
    cls_dir = os.path.join(DATASET, folder)
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f'  {cls}: {len(files)} images')
    for fname in files:
        try:
            X.append(extract(os.path.join(cls_dir, fname)))
            y.append(cls)
        except Exception as e:
            print(f'    Skip {fname}: {e}')

X = np.array(X)
print(f'\nTotal: {len(X)} images, {X.shape[1]} features')

le = LabelEncoder()
y_enc = le.fit_transform(y)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(X_sc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f'Train: {len(Xtr)}  Test: {len(Xte)}\n')

print('Training HistGradientBoosting...')
model = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_depth=6,
    min_samples_leaf=20, l2_regularization=0.1,
    random_state=42, class_weight='balanced'
)
model.fit(Xtr, ytr)

acc = accuracy_score(yte, model.predict(Xte))
print(f'\nAccuracy: {acc*100:.2f}%')
print(classification_report(yte, model.predict(Xte), target_names=le.classes_))

out = os.path.join(OUT_DIR, 'light_model.pkl')
with open(out, 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'le': le}, f)
size_mb = os.path.getsize(out) / 1024 / 1024
print(f'Saved: {out}  ({size_mb:.2f} MB)')
print(f'Classes: {list(le.classes_)}')
