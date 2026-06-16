from flask import Blueprint
from middleware.auth import jwt_required
from controllers.disease_controller import predict, get_history, get_history_by_id, delete_history

disease_bp = Blueprint('disease', __name__)

disease_bp.post('/predict')(jwt_required(predict))
disease_bp.get('/history')(jwt_required(get_history))
disease_bp.get('/history/<detection_id>')(jwt_required(get_history_by_id))
disease_bp.delete('/history/<detection_id>')(jwt_required(delete_history))
