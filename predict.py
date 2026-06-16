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

_ALGAL_IDX   = CLASS_NAMES.index('Tea Algal Leaf Spot')
_HEALTHY_IDX = CLASS_NAMES.index('Healthy')

_ort_session = None
_svm = None
_scaler = None


def _load_models():
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


def _extract_features(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)
    inp_name = _ort_session.get_inputs()[0].name
    features = _ort_session.run(None, {inp_name: arr})[0]
    return _scaler.transform(features)


def _apply_bias_correction(proba: np.ndarray, idx: int, confidence: float):
    """
    Tea Algal Leaf Spot and Healthy share similar visual features.
    The SVM tends to over-predict Tea Algal Leaf Spot on healthy leaves.
    Require >= 65% confidence to accept Tea Algal Leaf Spot; otherwise
    fall back to the next-best class.
    """
    if idx == _ALGAL_IDX and confidence < 0.65:
        sorted_idx = np.argsort(proba)[::-1]
        for fallback in sorted_idx:
            if int(fallback) != _ALGAL_IDX:
                return int(fallback), float(proba[fallback])
    return idx, confidence


def predict_from_bytes(image_bytes: bytes) -> dict:
    if not _load_models():
        return _demo_response(image_bytes)

    features   = _extract_features(image_bytes)
    proba      = _svm.predict_proba(features)[0]

    idx        = int(np.argmax(proba))
    confidence = float(proba[idx])

    idx, confidence = _apply_bias_correction(proba, idx, confidence)

    rejected = confidence < CONFIDENCE_THRESHOLD

    all_predictions = sorted(
        [{'disease': CLASS_NAMES[i], 'probability': float(p)} for i, p in enumerate(proba)],
        key=lambda x: x['probability'],
        reverse=True,
    )

    return {
        'disease':         CLASS_NAMES[idx],
        'confidence':      confidence,
        'all_predictions': all_predictions,
        'rejected':        rejected,
    }


def _demo_response(image_bytes: bytes) -> dict:
    seed  = int.from_bytes(image_bytes[:8], 'big') % (2 ** 31)
    proba = np.random.default_rng(seed).dirichlet(np.ones(len(CLASS_NAMES)) * 0.5)
    idx   = int(np.argmax(proba))
    idx, conf = _apply_bias_correction(proba, idx, float(proba[idx]))
    return {
        'disease':         CLASS_NAMES[idx],
        'confidence':      conf,
        'all_predictions': sorted(
            [{'disease': CLASS_NAMES[i], 'probability': float(p)} for i, p in enumerate(proba)],
            key=lambda x: x['probability'], reverse=True,
        ),
        'rejected': conf < CONFIDENCE_THRESHOLD,
    }
