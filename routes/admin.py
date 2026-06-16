from flask import Blueprint
from middleware.admin import admin_required
from controllers.admin_controller import (
    get_all_users, get_user, update_user_role,
    delete_user, get_admin_stats, get_all_scans, get_report,
)

admin_bp = Blueprint('admin', __name__)

admin_bp.get('/stats')(admin_required(get_admin_stats))
admin_bp.get('/users')(admin_required(get_all_users))
admin_bp.get('/users/<user_id>')(admin_required(get_user))
admin_bp.put('/users/<user_id>/role')(admin_required(update_user_role))
admin_bp.delete('/users/<user_id>')(admin_required(delete_user))
admin_bp.get('/scans')(admin_required(get_all_scans))
admin_bp.get('/reports')(admin_required(get_report))
