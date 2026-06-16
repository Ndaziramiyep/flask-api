import secrets
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

import bcrypt
from bson import ObjectId
from flask import request, jsonify, current_app
from flask_jwt_extended import create_access_token

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


def _send_email(to_email: str, subject: str, html_body: str):
    cfg = current_app.config
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = cfg['EMAIL_FROM']
    msg['To'] = to_email
    msg.attach(MIMEText(html_body, 'html'))

    with smtplib.SMTP(cfg['EMAIL_HOST'], cfg['EMAIL_PORT']) as server:
        server.ehlo()
        server.starttls()
        server.login(cfg['EMAIL_USER'], cfg['EMAIL_PASS'])
        server.sendmail(cfg['EMAIL_USER'], to_email, msg.as_string())


def register():
    data = request.get_json()
    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not name or len(name) < 2:
        return jsonify({'success': False, 'message': 'Name must be at least 2 characters'}), 400
    if not email or '@' not in email:
        return jsonify({'success': False, 'message': 'Please provide a valid email'}), 400
    if len(password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters'}), 400

    if users_col.find_one({'email': email}):
        return jsonify({'success': False, 'message': 'Email already registered'}), 400

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(12))
    now = datetime.utcnow()

    user_doc = {
        'name': name,
        'email': email,
        'password': hashed_pw,
        'avatar': None,
        'role': 'user',
        'notificationsEnabled': True,
        'resetPasswordToken': None,
        'resetPasswordExpire': None,
        'createdAt': now,
        'updatedAt': now,
    }

    result = users_col.insert_one(user_doc)
    user_doc['_id'] = result.inserted_id

    token = create_access_token(identity=str(result.inserted_id))

    try:
        _send_email(
            email,
            'Welcome to TeaScan AI!',
            f'<h2>Welcome, {name}!</h2><p>Your TeaScan AI account is ready. Start scanning tea leaves for disease detection.</p>'
        )
    except Exception:
        pass

    return jsonify({
        'success': True,
        'message': 'Account created successfully',
        'token': token,
        'user': _safe_user(user_doc),
    }), 201


def login():
    data = request.get_json()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password are required'}), 400

    user = users_col.find_one({'email': email})
    if not user:
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    stored_pw = user['password']
    if isinstance(stored_pw, str):
        stored_pw = stored_pw.encode('utf-8')

    if not bcrypt.checkpw(password.encode('utf-8'), stored_pw):
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    token = create_access_token(identity=str(user['_id']))
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'token': token,
        'user': _safe_user(user),
    }), 200


def forgot_password():
    data = request.get_json()
    email = (data.get('email') or '').strip().lower()

    generic_response = jsonify({
        'success': True,
        'message': 'If that email is registered, a reset link has been sent',
    }), 200

    user = users_col.find_one({'email': email})
    if not user:
        return generic_response

    raw_token = secrets.token_hex(20)
    hashed_token = hashlib.sha256(raw_token.encode()).hexdigest()
    expire = datetime.utcnow() + timedelta(minutes=10)

    users_col.update_one(
        {'_id': user['_id']},
        {'$set': {'resetPasswordToken': hashed_token, 'resetPasswordExpire': expire}}
    )

    reset_url = f"teascan://reset-password/{raw_token}"
    html = f"""
    <h2>Hi {user['name']},</h2>
    <p>You requested a password reset. Use this token in the app (expires in 10 minutes):</p>
    <p><strong style="font-size:18px">{raw_token}</strong></p>
    <p>If you didn't request this, ignore this email.</p>
    """
    try:
        _send_email(email, 'TeaScan AI - Password Reset', html)
    except Exception:
        users_col.update_one(
            {'_id': user['_id']},
            {'$set': {'resetPasswordToken': None, 'resetPasswordExpire': None}}
        )
        return jsonify({'success': False, 'message': 'Failed to send email. Try again later.'}), 500

    return generic_response


def reset_password():
    data = request.get_json()
    raw_token = (data.get('token') or '').strip()
    new_password = data.get('password') or ''

    if not raw_token:
        return jsonify({'success': False, 'message': 'Reset token is required'}), 400
    if len(new_password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters'}), 400

    hashed_token = hashlib.sha256(raw_token.encode()).hexdigest()
    user = users_col.find_one({
        'resetPasswordToken': hashed_token,
        'resetPasswordExpire': {'$gt': datetime.utcnow()},
    })

    if not user:
        return jsonify({'success': False, 'message': 'Invalid or expired reset token'}), 400

    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt(12))
    users_col.update_one(
        {'_id': user['_id']},
        {'$set': {
            'password': hashed_pw,
            'resetPasswordToken': None,
            'resetPasswordExpire': None,
            'updatedAt': datetime.utcnow(),
        }}
    )

    token = create_access_token(identity=str(user['_id']))
    return jsonify({'success': True, 'message': 'Password reset successfully', 'token': token}), 200
