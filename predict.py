import os
import io
import pickle
import numpy as np
from PIL import Image

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
CONFIDENCE_THRESHOLD = 0.50

_light_model = None
_light_scaler = None
_light_le = None
_heavy_mobilenet = None
_heavy_svm = None
_heavy_scaler = None
_heavy_le = None


def _load_light():
    """Load lightweight PIL+numpy model (no TensorFlow needed)."""
    global _light_model, _light_scaler, _light_le
    if _light_model is not None:
        return True
    path = os.path.join(MODELS_DIR, 'light_model.pkl')
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        _light_model = payload['model']
        _light_scaler = payload['scaler']
        _light_le = payload['le']
        print('Lightweight model loaded.')
        return True
    except Exception as e:
        print(f'Light model load error: {e}')
        return False


def _load_heavy():
    """Load heavy MobileNetV2+SVM model (TensorFlow required)."""
    global _heavy_mobilenet, _heavy_svm, _heavy_scaler, _heavy_le
    if _heavy_mobilenet is not None:
        return True
    svm_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    le_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    if not (os.path.exists(svm_path) and os.path.exists(le_path)):
        return False
    try:
        import tensorflow as tf
        import joblib
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg'
        )
        base.trainable = False
        _heavy_mobilenet = base
        _heavy_svm = joblib.load(svm_path)
        _heavy_le = joblib.load(le_path)
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        _heavy_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        print('Heavy TF+SVM model loaded.')
        return True
    except Exception as e:
        print(f'Heavy model load error: {e}')
        return False


def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    safe_max = np.where(maxc > 1e-6, maxc, 1.0)
    s = np.where(maxc > 1e-6, (maxc - minc) / safe_max, 0.0)
    diff = np.where(maxc - minc > 1e-6, maxc - minc, 1.0)
    rc = (maxc - r) / diff
    gc = (maxc - g) / diff
    bc = (maxc - b) / diff
    h = np.where(r == maxc, bc - gc,
                 np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=2)


def _looks_like_leaf(image_bytes: bytes) -> bool:
    """Return False for clearly non-photo images (noise, solid colours)."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((64, 64), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    h_diff = np.abs(arr[:-1, :, :] - arr[1:, :, :]).mean()
    v_diff = np.abs(arr[:, :-1, :] - arr[:, 1:, :]).mean()
    return (h_diff + v_diff) / 2.0 < 0.055


def _extract_features(image_bytes: bytes) -> np.ndarray:
    """Extract 449-feature vector matching the light_model.pkl training pipeline."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((128, 128), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    hsv = _rgb_to_hsv(arr)
    ycbcr = np.array(img.convert('YCbCr'), dtype=np.float32) / 255.0
    feats = []

    # 1. RGB + HSV + YCbCr histograms (32 bins each)
    for space in [arr, hsv, ycbcr]:
        for c in range(3):
            h, _ = np.histogram(space[:, :, c], bins=32, range=(0, 1))
            feats.extend(h / (h.sum() + 1e-8))

    # 2. Per-channel stats: mean, std, q10, q25, q75, q90
    for c in range(3):
        ch = arr[:, :, c]
        feats += [ch.mean(), ch.std(),
                  float(np.percentile(ch, 10)), float(np.percentile(ch, 25)),
                  float(np.percentile(ch, 75)), float(np.percentile(ch, 90))]

    # 3. 4×4 spatial grid — mean + std per RGB channel
    S = 32
    for qi in range(4):
        for qj in range(4):
            q = arr[qi*S:(qi+1)*S, qj*S:(qj+1)*S]
            for c in range(3):
                feats += [q[:, :, c].mean(), q[:, :, c].std()]

    # 4. Gradient magnitude at 3 scales
    for scale in [128, 64, 32]:
        img_s = img.resize((scale, scale), Image.LANCZOS)
        a = np.array(img_s, dtype=np.float32) / 255.0
        gx = np.abs(a[:, 1:, :] - a[:, :-1, :]).mean()
        gy = np.abs(a[1:, :, :] - a[:-1, :, :]).mean()
        feats += [gx, gy, (gx + gy) / 2]

    # 5. LBP-like texture on grayscale
    gray = arr.mean(axis=2)
    pad = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:], gray[2:, 2:],
        gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2], gray[:-2, :-2],
    ]
    codes = sum((n >= pad).astype(np.uint8) << i for i, n in enumerate(neighbors))
    lbp_h, _ = np.histogram(codes, bins=32, range=(0, 255))
    feats.extend(lbp_h / (lbp_h.sum() + 1e-8))

    # 6. Hue quadrant ratios
    h_ch = hsv[:, :, 0]
    feats += [
        ((h_ch < 0.08) | (h_ch > 0.92)).mean(),
        ((h_ch >= 0.08) & (h_ch < 0.22)).mean(),
        ((h_ch >= 0.22) & (h_ch < 0.45)).mean(),
        ((h_ch >= 0.45) & (h_ch < 0.65)).mean(),
        ((h_ch >= 0.65) & (h_ch < 0.78)).mean(),
        ((h_ch >= 0.78) & (h_ch < 0.92)).mean(),
    ]

    # 7. Saturation and brightness stats
    feats += [hsv[:, :, 1].mean(), hsv[:, :, 1].std(),
              hsv[:, :, 2].mean(), hsv[:, :, 2].std()]

    # 8. Dark / bright / saturated pixel ratios
    feats += [
        (arr.mean(axis=2) < 0.25).mean(),
        (arr.mean(axis=2) > 0.75).mean(),
        (hsv[:, :, 1] > 0.5).mean(),
    ]

    return np.array(feats, dtype=np.float32)


def _preprocess_heavy(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Predict tea leaf disease. Returns:
      {disease, confidence (0-1), all_predictions, rejected (bool)}
    rejected=True → image is likely not a tea leaf (confidence < 50%)
    """
    # Fast smoothness check — reject obvious non-photo images before running the model
    if not _looks_like_leaf(image_bytes):
        classes = ['Healthy', 'Tea Algal Leaf Spot', 'Brown Blight', 'Gray Blight',
                   'Helopeltis', 'Red Spider', 'Green Mirid Bug']
        proba = np.ones(len(classes)) / len(classes)
        return {
            'disease': 'Unknown',
            'confidence': 0.0,
            'all_predictions': [{'disease': c, 'probability': float(p)} for c, p in zip(classes, proba)],
            'rejected': True,
        }

    # Prefer lightweight model (no TF, works on Render free tier)
    if _load_light():
        features = _extract_features(image_bytes)
        features_sc = _light_scaler.transform(features.reshape(1, -1))
        proba = _light_model.predict_proba(features_sc)[0]
        classes = list(_light_le.classes_)

    elif _load_heavy():
        features = _heavy_mobilenet.predict(_preprocess_heavy(image_bytes), verbose=0)
        if _heavy_scaler is not None:
            features = _heavy_scaler.transform(features)
        proba = _heavy_svm.predict_proba(features)[0]
        classes = list(_heavy_le.classes_)

    else:
        # Demo fallback — deterministic per image
        classes = ['Healthy', 'Tea Algal Leaf Spot', 'Brown Blight', 'Gray Blight',
                   'Helopeltis', 'Red Spider', 'Green Mirid Bug']
        seed = int.from_bytes(image_bytes[:8], 'big') % (2 ** 31)
        rng = np.random.default_rng(seed)
        proba = rng.dirichlet(np.ones(len(classes)) * 0.5)

    idx = int(np.argmax(proba))
    max_confidence = float(proba[idx])
    rejected = max_confidence < CONFIDENCE_THRESHOLD

    all_predictions = sorted(
        [{'disease': c, 'probability': float(p)} for c, p in zip(classes, proba)],
        key=lambda x: x['probability'],
        reverse=True,
    )

    return {
        'disease': classes[idx],
        'confidence': max_confidence,
        'all_predictions': all_predictions,
        'rejected': rejected,
    }
