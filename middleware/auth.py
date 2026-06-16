from functools import wraps
from flask import jsonify
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
from bson import ObjectId
from config.database import users_col


def jwt_required(f):
    """Decorator that verifies JWT and injects current_user into kwargs."""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            verify_jwt_in_request()
            user_id = get_jwt_identity()
            user = users_col.find_one({'_id': ObjectId(user_id)})
            if not user:
                return jsonify({'success': False, 'message': 'User not found'}), 401
            kwargs['current_user'] = user
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'success': False, 'message': 'Token is invalid or expired'}), 401
    return decorated
