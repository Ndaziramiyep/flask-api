import os
from datetime import datetime

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
from flask_swagger_ui import get_swaggerui_blueprint

from config.settings import Config
from config.database import ping_db, create_indexes
from routes.auth import auth_bp
from routes.user import user_bp
from routes.disease import disease_bp
from routes.dashboard import dashboard_bp
from routes.notifications import notifications_bp
from routes.admin import admin_bp

load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    # Extend config with email settings
    app.config['EMAIL_HOST'] = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
    app.config['EMAIL_PORT'] = int(os.getenv('EMAIL_PORT', 587))
    app.config['EMAIL_USER'] = os.getenv('EMAIL_USER', '')
    app.config['EMAIL_PASS'] = os.getenv('EMAIL_PASS', '')
    app.config['EMAIL_FROM'] = os.getenv('EMAIL_FROM', 'TeaScan AI <noreply@teascan.ai>')

    # Extensions
    CORS(app, origins='*', supports_credentials=True)
    JWTManager(app)

    # Connect to MongoDB Atlas and set up indexes
    ping_db()
    create_indexes()

    # Ensure uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Serve uploaded images
    @app.route('/uploads/<path:filename>')
    def serve_upload(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    # Health check
    @app.route('/')
    @app.route('/health')
    def health():
        return jsonify({
            'success': True,
            'service': 'TeaScan AI Flask API',
            'timestamp': datetime.utcnow().isoformat(),
        }), 200

    # Swagger UI
    swaggerui_bp = get_swaggerui_blueprint(
        '/api/docs',
        '/static/swagger.json',
        config={'app_name': 'TeaScan AI API'},
    )
    app.register_blueprint(swaggerui_bp, url_prefix='/api/docs')

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(user_bp, url_prefix='/api/user')
    app.register_blueprint(disease_bp, url_prefix='/api/disease')
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    app.register_blueprint(notifications_bp, url_prefix='/api/notifications')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')

    # Global error handlers
    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({'success': False, 'message': 'Route not found'}), 404

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({'success': False, 'message': 'File too large (max 10MB)'}), 413

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'success': False, 'message': str(e)}), 500

    return app


app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    print(f'TeaScan AI Flask API starting on port {port}')
    print(f'Health check: http://localhost:{port}/health')
    app.run(host='0.0.0.0', port=port, debug=debug)
