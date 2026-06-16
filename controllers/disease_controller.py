import os
import sys
import uuid
from datetime import datetime

from flask import request, jsonify, current_app
from bson import ObjectId

from config.database import detections_col, notifications_col


def _get_severity(disease: str, confidence: float) -> str:
    if disease == 'Healthy':
        return 'None'
    high_risk = {'Helopeltis', 'Green Mirid Bug', 'Red Spider'}
    if confidence >= 80:
        return 'High' if disease in high_risk else 'Medium'
    elif confidence >= 50:
        return 'Medium' if disease in high_risk else 'Low'
    return 'Low'


def _allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'webp'}


def _serialize(doc: dict) -> dict:
    doc = dict(doc)
    doc['_id'] = str(doc['_id'])
    for field in ('createdAt', 'updatedAt'):
        if field in doc and hasattr(doc[field], 'isoformat'):
            doc[field] = doc[field].isoformat()
    return doc


def _load_predictor():
    """Lazy-load the AI prediction function from ai-models/."""
    ai_models_dir = current_app.config['AI_MODELS_DIR']
    if ai_models_dir not in sys.path:
        sys.path.insert(0, ai_models_dir)
    from predict import predict_from_bytes  # noqa
    return predict_from_bytes


def predict(current_user):
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided'}), 400

    file = request.files['image']
    if not file or not _allowed_file(file.filename or ''):
        return jsonify({'success': False, 'message': 'Invalid file. Use JPG, PNG, or WebP'}), 400

    # Save image to uploads/
    ext = (file.filename or 'jpg').rsplit('.', 1)[-1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, filename)
    image_bytes = file.read()

    with open(file_path, 'wb') as f:
        f.write(image_bytes)

    # Run AI prediction
    try:
        predict_from_bytes = _load_predictor()
        result = predict_from_bytes(image_bytes)
    except Exception as e:
        os.remove(file_path)
        return jsonify({'success': False, 'message': f'AI prediction failed: {str(e)}'}), 500

    # Reject non-tea-leaf images
    if result.get('rejected'):
        os.remove(file_path)
        return jsonify({
            'success': False,
            'message': 'Uncertain image. Please try with a clearer photo of a tea leaf.',
        }), 422

    disease = result['disease']
    confidence_pct = round(result['confidence'] * 100, 2)
    all_predictions_pct = [
        {'disease': p['disease'], 'probability': round(p['probability'] * 100, 2)}
        for p in result['all_predictions']
    ]
    severity = _get_severity(disease, confidence_pct)
    now = datetime.utcnow()

    image_path = f"uploads/{filename}"
    detection_doc = {
        'userId': str(current_user['_id']),
        'imagePath': image_path,
        'disease': disease,
        'confidence': confidence_pct,
        'severity': severity,
        'allPredictions': all_predictions_pct,
        'notes': request.form.get('notes'),
        'location': request.form.get('location'),
        'createdAt': now,
        'updatedAt': now,
    }

    inserted = detections_col.insert_one(detection_doc)
    detection_doc['_id'] = inserted.inserted_id

    # Create notification for disease findings
    if disease != 'Healthy' and current_user.get('notificationsEnabled', True):
        notifications_col.insert_one({
            'userId': str(current_user['_id']),
            'title': f'Disease Detected: {disease}',
            'message': f'{disease} detected with {confidence_pct:.1f}% confidence. Severity: {severity}.',
            'type': 'warning',
            'isRead': False,
            'createdAt': now,
        })

    return jsonify({
        'success': True,
        'message': 'Image analyzed successfully',
        'data': _serialize(detection_doc),
    }), 200


def get_history(current_user):
    page = max(1, int(request.args.get('page', 1)))
    limit = min(50, max(1, int(request.args.get('limit', 10))))
    skip = (page - 1) * limit

    query = {'userId': str(current_user['_id'])}
    if request.args.get('disease'):
        query['disease'] = request.args.get('disease')

    total = detections_col.count_documents(query)
    records = list(detections_col.find(query).sort('createdAt', -1).skip(skip).limit(limit))

    return jsonify({
        'success': True,
        'data': [_serialize(r) for r in records],
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'pages': (total + limit - 1) // limit,
            'hasNext': page < (total + limit - 1) // limit,
            'hasPrev': page > 1,
        },
    }), 200


def get_history_by_id(current_user, detection_id):
    try:
        record = detections_col.find_one({'_id': ObjectId(detection_id), 'userId': str(current_user['_id'])})
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid ID'}), 400

    if not record:
        return jsonify({'success': False, 'message': 'Scan not found'}), 404

    return jsonify({'success': True, 'data': _serialize(record)}), 200


def delete_history(current_user, detection_id):
    try:
        record = detections_col.find_one({'_id': ObjectId(detection_id), 'userId': str(current_user['_id'])})
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid ID'}), 400

    if not record:
        return jsonify({'success': False, 'message': 'Scan not found'}), 404

    # Remove image file
    upload_folder = current_app.config['UPLOAD_FOLDER']
    file_path = os.path.join(upload_folder, '..', record['imagePath'])
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path):
        os.remove(file_path)

    detections_col.delete_one({'_id': record['_id']})
    return jsonify({'success': True, 'message': 'Scan deleted successfully'}), 200
