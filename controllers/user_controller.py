import os
from datetime import datetime

import bcrypt
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename

from config.database import users_col


def _safe_user(user: dict) -> dict:
    return {
        '_id': str(user['_id']),
        'name': user['name'],
        'email': user['email'],
        'avatar': user.get('avatar'),
        'role': user.get('role', 'user'),
        'notificationsEnabled': user.get('notificationsEnabled', True),
        'createdAt': user.get('createdAt', datetime.utcnow()).isoformat(),
    }


def _allowed_file(filename: str) -> bool:
    allowed = {'jpg', 'jpeg', 'png', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def get_profile(current_user):
    return jsonify({'success': True, 'data': _safe_user(current_user)}), 200


def update_profile(current_user):
    updates = {}

    json_data = request.get_json(silent=True) or {}
    name = request.form.get('name') or json_data.get('name')
    if name:
        updates['name'] = name.strip()

    notifications_enabled = json_data.get('notificationsEnabled')
    if notifications_enabled is None:
        val = request.form.get('notificationsEnabled')
        if val is not None:
            notifications_enabled = val.lower() in ('true', '1', 'yes')
    if notifications_enabled is not None:
        updates['notificationsEnabled'] = notifications_enabled

    if 'avatar' in request.files:
        file = request.files['avatar']
        if file and _allowed_file(file.filename or ''):
            import uuid
            ext = (file.filename or 'jpg').rsplit('.', 1)[-1].lower()
            filename = f"avatar_{str(current_user['_id'])}_{uuid.uuid4().hex}.{ext}"
            save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            updates['avatar'] = f"uploads/{filename}"

    if not updates:
        return jsonify({'success': False, 'message': 'No fields to update'}), 400

    updates['updatedAt'] = datetime.utcnow()
    users_col.update_one({'_id': current_user['_id']}, {'$set': updates})
    updated = users_col.find_one({'_id': current_user['_id']})

    return jsonify({'success': True, 'message': 'Profile updated', 'data': _safe_user(updated)}), 200


def change_password(current_user):
    data = request.get_json()
    current_password = data.get('currentPassword') or ''
    new_password = data.get('newPassword') or ''

    if not bcrypt.checkpw(current_password.encode('utf-8'), current_user['password']):
        return jsonify({'success': False, 'message': 'Current password is incorrect'}), 401

    if len(new_password) < 8:
        return jsonify({'success': False, 'message': 'New password must be at least 8 characters'}), 400

    hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt(12))
    users_col.update_one({'_id': current_user['_id']}, {'$set': {'password': hashed, 'updatedAt': datetime.utcnow()}})

    return jsonify({'success': True, 'message': 'Password changed successfully'}), 200


def delete_account(current_user):
    from config.database import detections_col, notifications_col
    detections_col.delete_many({'userId': str(current_user['_id'])})
    notifications_col.delete_many({'userId': str(current_user['_id'])})
    users_col.delete_one({'_id': current_user['_id']})
    return jsonify({'success': True, 'message': 'Account deleted successfully'}), 200
