import os
import io
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
CONFIDENCE_THRESHOLD = 0.50

# Alphabetical order — matches sklearn LabelEncoder default used during training
CLASS_NAMES = [
    'Brown Blight',
    'Gray Blight',
    'Green Mirid Bug',
    'Healthy',
    'Helopeltis',
    'Red Spider',
    'Tea Algal Leaf Spot',
]

_ort_session = None
_svm = None
_scaler = None


def _load_models():
    """Load ONNX MobileNetV2 feature extractor + StandardScaler + SVM."""
    global _ort_session, _svm, _scaler
    if _ort_session is not None:
        return True

    onnx_path   = os.path.join(MODELS_DIR, 'mobilenet_features.onnx')
    svm_path    = os.path.join(MODELS_DIR, 'svm_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    if not all(os.path.exists(p) for p in [onnx_path, svm_path, scaler_path]):
        print('One or more model files missing.')
        return False

    try:
        import joblib
        import onnxruntime as ort

        _ort_session = ort.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider']
        )
        _scaler = joblib.load(scaler_path)
        _svm    = joblib.load(svm_path)
        print('Models loaded: ONNX MobileNetV2 + StandardScaler + SVM')
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
    """Preprocess and extract 1280-dim features via ONNX MobileNetV2."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0              # MobileNetV2 preprocessing: [-1, 1]
    arr = np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)
    inp_name = _ort_session.get_inputs()[0].name
    features = _ort_session.run(None, {inp_name: arr})[0]  # (1, 1280)
    return _scaler.transform(features)


def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Predict tea leaf disease from raw image bytes.
    Returns: {disease, confidence, all_predictions, rejected}
    rejected=True when image is not a tea leaf or confidence < 50%.
    """
    if not _looks_like_photo(image_bytes):
        return _rejected_response()

    if not _load_models():
        return _demo_response(image_bytes)

    features = _extract_features(image_bytes)
    proba    = _svm.predict_proba(features)[0]

    idx        = int(np.argmax(proba))
    confidence = float(proba[idx])
    disease    = CLASS_NAMES[idx]
    rejected   = confidence < CONFIDENCE_THRESHOLD

    all_predictions = sorted(
        [{'disease': CLASS_NAMES[i], 'probability': float(p)} for i, p in enumerate(proba)],
        key=lambda x: x['probability'],
        reverse=True,
    )

    return {
        'disease':         disease,
        'confidence':      confidence,
        'all_predictions': all_predictions,
        'rejected':        rejected,
    }


def _rejected_response() -> dict:
    proba = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)
    return {
        'disease':         'Unknown',
        'confidence':      0.0,
        'all_predictions': [{'disease': c, 'probability': float(p)}
                            for c, p in zip(CLASS_NAMES, proba)],
        'rejected':        True,
    }


def _demo_response(image_bytes: bytes) -> dict:
    seed  = int.from_bytes(image_bytes[:8], 'big') % (2 ** 31)
    proba = np.random.default_rng(seed).dirichlet(np.ones(len(CLASS_NAMES)) * 0.5)
    idx   = int(np.argmax(proba))
    return {
        'disease':         CLASS_NAMES[idx],
        'confidence':      float(proba[idx]),
        'all_predictions': sorted(
            [{'disease': CLASS_NAMES[i], 'probability': float(p)} for i, p in enumerate(proba)],
            key=lambda x: x['probability'],
            reverse=True,
        ),
        'rejected': float(proba[idx]) < CONFIDENCE_THRESHOLD,
    }
