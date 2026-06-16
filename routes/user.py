from flask import Blueprint
from middleware.auth import jwt_required
from controllers.user_controller import get_profile, update_profile, change_password, delete_account

user_bp = Blueprint('user', __name__)

user_bp.get('/profile')(jwt_required(get_profile))
user_bp.put('/profile')(jwt_required(update_profile))
user_bp.put('/change-password')(jwt_required(change_password))
user_bp.delete('/account')(jwt_required(delete_account))
