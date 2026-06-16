"""
Entry point untuk RAG Perdes API Server.

Usage:
  Development:  python run.py
  Production:   gunicorn -w 1 -b 0.0.0.0:6000 run:app --timeout 600

The server binds to 0.0.0.0 so it's accessible from external machines.
Port 6000 is the default; change via PORT env variable.
"""

import os
import logging
from app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6000))
    logging.info(f"Starting RAG Perdes API on 0.0.0.0:{port}")
    logging.info(f"External access: http://<server-ip>:{port}/api/health")
    app.run(host="0.0.0.0", port=port, debug=False)
