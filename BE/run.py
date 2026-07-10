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
    logging.info(f"Starting RAFT API on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)