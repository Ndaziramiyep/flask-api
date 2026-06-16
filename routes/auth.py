from flask import Blueprint
from controllers.auth_controller import register, login, forgot_password, reset_password

auth_bp = Blueprint('auth', __name__)

auth_bp.post('/register')(register)
auth_bp.post('/login')(login)
auth_bp.post('/forgot-password')(forgot_password)
auth_bp.post('/reset-password')(reset_password)
