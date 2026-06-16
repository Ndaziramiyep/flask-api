from flask import Blueprint
from middleware.auth import jwt_required
from controllers.dashboard_controller import get_stats

dashboard_bp = Blueprint('dashboard', __name__)
dashboard_bp.get('/stats')(jwt_required(get_stats))
