"""
API Routes for RAFT Inference Service.

Provides endpoints for plain chat, RAG/RAFT chat with documents,
model management, and health checks.
"""

import os
import logging
from flask import request, jsonify
from llama_service import LlamaService

logger = logging.getLogger(__name__)


def register_routes(app):
    """
    Register all API routes with the Flask app.

    Args:
        app: Flask application instance.
    """
    service = LlamaService.get_instance()

    # -------------------------------------------------------------------------
    # POST /api/chat — Plain LLM chat without documents
    # -------------------------------------------------------------------------
    @app.route('/api/chat', methods=['POST'])
    def chat():
        """Plain chat endpoint without document context."""
        try:
            # Check if model is loaded
            if service.model is None:
                return jsonify({
                    "status": "error",
                    "message": "No model loaded. Use POST /api/load-model first."
                }), 503

            # Parse request body
            data = request.get_json()
            if not data:
                return jsonify({
                    "status": "error",
                    "message": "Request body must be JSON"
                }), 400

            # Validate pertanyaan
            pertanyaan = data.get('pertanyaan')
            if not pertanyaan or not pertanyaan.strip():
                return jsonify({
                    "status": "error",
                    "message": "Field 'pertanyaan' is required and must be non-empty"
                }), 400

            # Optional parameters
            max_tokens = data.get('max_tokens', 512)
            temperature = data.get('temperature', 0.7)
            top_p = data.get('top_p', 0.9)

            # Generate answer
            result = service.generate_answer(
                pertanyaan=pertanyaan.strip(),
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p)
            )

            # Check for generation error
            if 'error' in result:
                return jsonify({
                    "status": "error",
                    "message": result['error']
                }), 500

            return jsonify({
                "status": "success",
                "pertanyaan": pertanyaan,
                "jawaban": result['raw_response'],
                "model_type": result['model_type']
            })

        except Exception as e:
            logger.error(f"Error in /api/chat: {e}")
            return jsonify({
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }), 500

    # -------------------------------------------------------------------------
    # POST /api/chat-rag — RAFT chat with document context
    # -------------------------------------------------------------------------
    @app.route('/api/chat-rag', methods=['POST'])
    def chat_rag():
        """RAG/RAFT chat endpoint with document context."""
        try:
            # Check if model is loaded
            if service.model is None:
                return jsonify({
                    "status": "error",
                    "message": "No model loaded. Use POST /api/load-model first."
                }), 503

            # Parse request body
            data = request.get_json()
            if not data:
                return jsonify({
                    "status": "error",
                    "message": "Request body must be JSON"
                }), 400

            # Validate pertanyaan
            pertanyaan = data.get('pertanyaan')
            if not pertanyaan or not pertanyaan.strip():
                return jsonify({
                    "status": "error",
                    "message": "Field 'pertanyaan' is required and must be non-empty"
                }), 400

            # Validate dokumen
            dokumen = data.get('dokumen')
            if not dokumen or not isinstance(dokumen, list) or len(dokumen) == 0:
                return jsonify({
                    "status": "error",
                    "message": "Field 'dokumen' is required and must be a non-empty list"
                }), 400

            # Optional parameters
            max_tokens = data.get('max_tokens', 512)
            temperature = data.get('temperature', 0.7)
            top_p = data.get('top_p', 0.9)

            # Generate RAG answer
            result = service.generate_answer_rag(
                pertanyaan=pertanyaan.strip(),
                dokumen=dokumen,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p)
            )

            # Check for generation error
            if 'error' in result:
                return jsonify({
                    "status": "error",
                    "message": result['error']
                }), 500

            return jsonify({
                "status": "success",
                "pertanyaan": pertanyaan,
                "analisis": result['analisis'],
                "jawaban": result['jawaban'],
                "raw_response": result['raw_response'],
                "model_type": result['model_type'],
                "num_documents": result['num_documents']
            })

        except Exception as e:
            logger.error(f"Error in /api/chat-rag: {e}")
            return jsonify({
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }), 500

    # -------------------------------------------------------------------------
    # POST /api/load-model — Load or switch model
    # -------------------------------------------------------------------------
    @app.route('/api/load-model', methods=['POST'])
    def load_model():
        """Load or switch the active model."""
        try:
            # Parse request body
            data = request.get_json()
            if not data:
                return jsonify({
                    "status": "error",
                    "message": "Request body must be JSON"
                }), 400

            # Validate model_path
            model_path = data.get('model_path')
            if not model_path or not model_path.strip():
                return jsonify({
                    "status": "error",
                    "message": "Field 'model_path' is required"
                }), 400

            model_path = model_path.strip()

            # Check if path exists
            if not os.path.exists(model_path):
                return jsonify({
                    "status": "error",
                    "message": f"Model path does not exist: {model_path}"
                }), 400

            # Load model
            service.load_model(model_path)

            return jsonify({
                "status": "success",
                "model_path": service.model_path,
                "model_type": service.model_type
            })

        except RuntimeError as e:
            logger.error(f"Model loading failed: {e}")
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500

        except Exception as e:
            logger.error(f"Error in /api/load-model: {e}")
            return jsonify({
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }), 500

    # -------------------------------------------------------------------------
    # GET /api/model-info — Info about currently loaded model
    # -------------------------------------------------------------------------
    @app.route('/api/model-info', methods=['GET'])
    def model_info():
        """Get information about the currently loaded model."""
        try:
            info = service.get_model_info()
            return jsonify(info)

        except Exception as e:
            logger.error(f"Error in /api/model-info: {e}")
            return jsonify({
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }), 500

    # -------------------------------------------------------------------------
    # GET /api/models — List available models
    # -------------------------------------------------------------------------
    @app.route('/api/models', methods=['GET'])
    def list_models():
        """List available models in the model directory."""
        try:
            # Scan ../model/ directory relative to BE/
            model_base_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
            model_base_dir = os.path.abspath(model_base_dir)

            models = []

            if os.path.exists(model_base_dir):
                for entry in os.listdir(model_base_dir):
                    model_dir = os.path.join(model_base_dir, entry)

                    # Check if it's a directory with config.json
                    if os.path.isdir(model_dir):
                        config_path = os.path.join(model_dir, 'config.json')
                        if os.path.exists(config_path):
                            # Determine model type from name
                            model_type = "raft" if "raft" in entry.lower() else "base"
                            models.append({
                                "name": entry,
                                "path": model_dir,
                                "type": model_type
                            })

            return jsonify({
                "status": "success",
                "models": models
            })

        except Exception as e:
            logger.error(f"Error in /api/models: {e}")
            return jsonify({
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }), 500

    # -------------------------------------------------------------------------
    # GET /api/health — Health check
    # -------------------------------------------------------------------------
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        try:
            model_loaded = service.model is not None
            model_path = service.model_path

            endpoints = [
                "POST /api/chat",
                "POST /api/chat-rag",
                "POST /api/load-model",
                "GET /api/model-info",
                "GET /api/models",
                "GET /api/health"
            ]

            return jsonify({
                "status": "ok",
                "model_loaded": model_loaded,
                "model_path": model_path,
                "endpoints": endpoints
            })

        except Exception as e:
            logger.error(f"Error in /api/health: {e}")
            return jsonify({
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }), 500
