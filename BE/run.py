# """
# Entry point untuk RAG Perdes API Server.

# Usage:
#   Development:  python run.py
#   Production:   gunicorn -w 1 -b 0.0.0.0:6000 run:app --timeout 600

# The server binds to 0.0.0.0 so it's accessible from external machines.
# Port 6000 is the default; change via PORT env variable.
# """

# import os
# import logging
# from app import create_app

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )

# app = create_app()

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 6000))
#     logging.info(f"Starting RAG Perdes API on 0.0.0.0:{port}")
#     logging.info(f"External access: http://<server-ip>:{port}/api/health")
#     app.run(host="0.0.0.0", port=port, debug=False)

"""
RAFT Inference API - Entry Point.

Starts the Flask server for serving RAFT model inference.
Supports graceful shutdown and degraded mode (no model loaded).
"""

import os
import sys
import logging
import signal
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration from environment
MODEL_PATH = os.getenv("MODEL_PATH", "../model/raft_merged")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "6000"))

# Import after logging setup
from app import create_app
from llama_service import LlamaService


def main():
    """Main entry point - initialize and start the server."""
    app = create_app()
    service = LlamaService.get_instance()

    # Try loading model, but start server even if it fails (degraded mode)
    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Loading model from: {MODEL_PATH}")
            service.load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Server will start in degraded mode (no model loaded)")
    else:
        logger.warning(f"Model path not found: {MODEL_PATH}")
        logger.warning(
            "Server will start in degraded mode. "
            "Use POST /api/load-model to load a model."
        )

    # Graceful shutdown handler
    def signal_handler(sig, frame):
        logger.info("Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    logger.info(f"Starting server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT)


if __name__ == "__main__":
    main()
