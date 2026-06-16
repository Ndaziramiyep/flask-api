from flask import Blueprint
from middleware.auth import jwt_required
from controllers.notification_controller import (
    get_notifications, get_unread_count, mark_as_read, mark_all_as_read, delete_notification
)

notifications_bp = Blueprint('notifications', __name__)

notifications_bp.get('/')(jwt_required(get_notifications))
notifications_bp.get('/unread-count')(jwt_required(get_unread_count))
notifications_bp.put('/mark-all-read')(jwt_required(mark_all_as_read))
notifications_bp.put('/<notification_id>/read')(jwt_required(mark_as_read))
notifications_bp.delete('/<notification_id>')(jwt_required(delete_notification))
