from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    # Enable CORS for external access from any origin
    # This allows your RAG project (or any client) to call this API
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    from routes import bp
    app.register_blueprint(bp)

    return app
