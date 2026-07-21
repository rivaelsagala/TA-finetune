# from flask import Blueprint, request, jsonify
# from raft_service import load_model, generate_answer, generate_answer_without_context

# bp = Blueprint("routes", __name__)


# @bp.route("/api/health", methods=["GET"])
# def health():
#     return jsonify({
#         "status": "ok",
#         "message": "RAFT API is running"
#     }), 200


# @bp.route("/api/load-model", methods=["POST"])
# def load_model_endpoint():
#     try:
#         load_model()
#         return jsonify({
#             "status": "success",
#             "message": "Model berhasil dimuat ke memori."
#         }), 200
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500


# # @bp.route("/api/chat-raft", methods=["POST"])
# # def chat_rag():
# #     """
# #     Body JSON:
# #     {
# #       "question": "Sebutkan unsur masyarakat dalam Pasal 12 ayat (2) Perdes Majasetra?",
# #       "documents": [
# #         "dokumen 1 ...",
# #         "dokumen 2 ...",
# #         "dokumen 3 ..."
# #       ]
# #     }
# #     """
# #     try:
# #         data = request.get_json(force=True)

# #         question = (data.get("question") or "").strip()
# #         documents = data.get("documents", [])

# #         if not question:
# #             return jsonify({
# #                 "status": "error",
# #                 "message": "Field 'question' wajib diisi."
# #             }), 400

# #         if not isinstance(documents, list) or len(documents) == 0:
# #             return jsonify({
# #                 "status": "error",
# #                 "message": "Field 'documents' wajib berupa list dan tidak boleh kosong."
# #             }), 400

# #         result = generate_answer(
# #             question=question,
# #             documents=documents
# #         )
 
# #         return jsonify({
# #             "status": "success",
# #             "question": question,
# #             "documents_count": len(documents),
# #             "konteks_dipilih": result.get("konteks_dipilih", ""),
# #             "konteks_ditolak": result.get("konteks_ditolak", ""),
# #             "thought": result.get("thought_process", ""),
# #             "answer": result.get("jawaban", "")
# #         }), 200

# #     except Exception as e:
# #         return jsonify({
# #             "status": "error",
# #             "message": str(e)
# #         }), 500


# @bp.route("/api/chat-raft", methods=["POST"])
# def chat_without_context():
#     """
#     Endpoint untuk menguji model tanpa context.

#     Body JSON:
#     {
#         "question": "BPD itu apa sih, fungsinya buat apa di desa?"
#     }
#     """

#     try:
#         data = request.get_json(silent=True) or {}

#         question = str(data.get("question") or "").strip()

#         if not question:
#             return jsonify({
#                 "status": "error",
#                 "message": "Field 'question' wajib diisi."
#             }), 400

#         result = generate_answer_without_context(
#             question=question
#         )

#         return jsonify({
#             "status": "success",
#             "mode": "without_context",
#             "question": question,
#             "thought": result.get("thought_process", ""),
#             "answer": result.get("jawaban", ""),
#             "raw_output": result.get("raw_output", "")
#         }), 200

#     except ValueError as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 400

#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500