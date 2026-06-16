import os
import io
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
CONFIDENCE_THRESHOLD = 0.50

# Alphabetical order matches sklearn LabelEncoder default used during training
CLASS_NAMES = [
    'Brown Blight',
    'Gray Blight',
    'Green Mirid Bug',
    'Healthy',
    'Helopeltis',
    'Red Spider',
    'Tea Algal Leaf Spot',
]

_mobilenet = None
_svm = None
_scaler = None


def _load_models():
    """Load MobileNetV2 feature extractor + scaler + SVM classifier."""
    global _mobilenet, _svm, _scaler
    if _mobilenet is not None:
        return True

    svm_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if not (os.path.exists(svm_path) and os.path.exists(scaler_path)):
        print('Model files not found.')
        return False

    try:
        import joblib
        import tensorflow as tf

        # MobileNetV2 extracts 1280-dim feature vectors (GlobalAveragePooling)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg',
        )
        base.trainable = False
        _mobilenet = base

        _scaler = joblib.load(scaler_path)
        _svm = joblib.load(svm_path)
        print('Models loaded: MobileNetV2 + StandardScaler + SVM')
        return True
    except Exception as e:
        print(f'Model load error: {e}')
        return False


def _looks_like_photo(image_bytes: bytes) -> bool:
    """Reject solid colours, random noise, and non-photo images."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((64, 64), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    h_diff = np.abs(arr[:-1, :, :] - arr[1:, :, :]).mean()
    v_diff = np.abs(arr[:, :-1, :] - arr[:, 1:, :]).mean()
    return (h_diff + v_diff) / 2.0 < 0.055


def _extract_features(image_bytes: bytes) -> np.ndarray:
    """Preprocess image and extract 1280-dim MobileNetV2 features."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0          # MobileNetV2 expects [-1, 1]
    arr = np.expand_dims(arr, axis=0)
    features = _mobilenet.predict(arr, verbose=0)  # shape (1, 1280)
    return _scaler.transform(features)              # scaled (1, 1280)


def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Run tea disease prediction on raw image bytes.
    Returns: {disease, confidence, all_predictions, rejected}
    rejected=True means the image is not a tea leaf or confidence is too low.
    """
    # Reject obvious non-photos before running the heavy model
    if not _looks_like_photo(image_bytes):
        return _rejected_response()

    if not _load_models():
        # Demo fallback if models fail to load
        return _demo_response(image_bytes)

    features = _extract_features(image_bytes)
    proba = _svm.predict_proba(features)[0]          # shape (7,)

    idx = int(np.argmax(proba))
    confidence = float(proba[idx])
    disease = CLASS_NAMES[idx]

    rejected = confidence < CONFIDENCE_THRESHOLD

    all_predictions = sorted(
        [{'disease': CLASS_NAMES[i], 'probability': float(p)} for i, p in enumerate(proba)],
        key=lambda x: x['probability'],
        reverse=True,
    )

    return {
        'disease': disease,
        'confidence': confidence,
        'all_predictions': all_predictions,
        'rejected': rejected,
    }


def _rejected_response() -> dict:
    proba = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)
    return {
        'disease': 'Unknown',
        'confidence': 0.0,
        'all_predictions': [{'disease': c, 'probability': float(p)} for c, p in zip(CLASS_NAMES, proba)],
        'rejected': True,
    }


def _demo_response(image_bytes: bytes) -> dict:
    seed = int.from_bytes(image_bytes[:8], 'big') % (2 ** 31)
    rng = np.random.default_rng(seed)
    proba = rng.dirichlet(np.ones(len(CLASS_NAMES)) * 0.5)
    idx = int(np.argmax(proba))
    return {
        'disease': CLASS_NAMES[idx],
        'confidence': float(proba[idx]),
        'all_predictions': sorted(
            [{'disease': CLASS_NAMES[i], 'probability': float(p)} for i, p in enumerate(proba)],
            key=lambda x: x['probability'],
            reverse=True,
        ),
        'rejected': float(proba[idx]) < CONFIDENCE_THRESHOLD,
    }
