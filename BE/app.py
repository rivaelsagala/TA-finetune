"""
Flask Application Factory.

Creates and configures the Flask app with CORS, middleware,
error handlers, and registered routes.
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from routes import register_routes

logger = logging.getLogger(__name__)


def create_app():
    """
    Create and configure the Flask application.

    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__)

    # Configure CORS from environment
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    if cors_origins == "*":
        CORS(app, origins="*")
    else:
        CORS(app, origins=cors_origins.split(","))

    # Register API routes
    register_routes(app)

    # Request logging middleware
    @app.before_request
    def log_request():
        logger.info(f"{request.method} {request.path} - {request.remote_addr}")

    @app.after_request
    def log_response(response):
        logger.info(f"{request.method} {request.path} - {response.status_code}")
        return response

    # Error handlers
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"status": "error", "message": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status": "error", "message": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"status": "error", "message": "Internal server error"}), 500

    return app
