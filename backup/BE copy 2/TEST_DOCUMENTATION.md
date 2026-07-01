# API Test Documentation - RAFT Inference Service

Base URL: `http://localhost:6000`

---

## 1. GET /api/health — Health Check

Check server status, whether a model is loaded, and list all available endpoints.

### Request

```bash
curl -X GET http://localhost:6000/api/health
```

### Expected Response

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "../model/raft_merged",
  "endpoints": [
    "POST /api/chat",
    "POST /api/chat-rag",
    "POST /api/load-model",
    "GET /api/model-info",
    "GET /api/models",
    "GET /api/health"
  ]
}
```

### Notes
- `model_loaded` will be `false` if the server started in degraded mode (no model).
- Always call this first to verify the server is running.

---

## 2. GET /api/models — List Available Models

List all models found in the `model/` directory that contain a `config.json`.

### Request

```bash
curl -X GET http://localhost:6000/api/models
```

### Expected Response

```json
{
  "status": "success",
  "models": [
    {
      "name": "Meta-Llama-3.1-8B-Instruct",
      "path": "/workspace/model/Meta-Llama-3.1-8B-Instruct",
      "type": "base"
    },
    {
      "name": "raft_merged",
      "path": "/workspace/model/raft_merged",
      "type": "raft"
    }
  ]
}
```

### Notes
- `type` is `"raft"` if the folder name contains "raft", otherwise `"base"`.
- Use the `path` value as input for `/api/load-model`.

---

## 3. POST /api/load-model — Load or Switch Model

Load a model from a given path into GPU/CPU memory.

### Request

```bash
curl -X POST http://localhost:6000/api/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/workspace/model/raft_merged"
  }'
```

### Expected Response (Success)

```json
{
  "status": "success",
  "model_path": "/workspace/model/raft_merged",
  "model_type": "raft"
}
```

### Expected Response (Invalid Path)

```json
{
  "status": "error",
  "message": "Model path does not exist: /invalid/path"
}
```
HTTP Status: `400`

### Notes
- This will unload the current model and load the new one.
- Loading may take several minutes depending on model size.
- The server starts with the model from `MODEL_PATH` env variable (default: `../model/raft_merged`).

---

## 4. GET /api/model-info — Current Model Info

Get details about the currently loaded model, including GPU memory usage.

### Request

```bash
curl -X GET http://localhost:6000/api/model-info
```

### Expected Response (Model Loaded)

```json
{
  "model_path": "/workspace/model/raft_merged",
  "model_type": "raft",
  "loaded": true,
  "gpu_memory_allocated_gb": 15.23,
  "gpu_memory_reserved_gb": 16.0,
  "gpu_device_count": 1
}
```

### Expected Response (No Model)

```json
{
  "model_path": null,
  "model_type": null,
  "loaded": false,
  "gpu_memory_allocated_gb": 0,
  "gpu_memory_reserved_gb": 0,
  "gpu_device_count": 0
}
```

---

## 5. POST /api/chat — Plain Chat (No Documents)

Send a question to the model without any document context. Uses the RAFT system prompt but no retrieval augmentation.

### Request

```bash
curl -X POST http://localhost:6000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "pertanyaan": "Apa itu Peraturan Desa?",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### Request Parameters

| Field          | Type   | Required | Default | Description                        |
|----------------|--------|----------|---------|------------------------------------|
| `pertanyaan`   | string | Yes      | —       | The question to ask the model      |
| `max_tokens`   | int    | No       | 512     | Maximum tokens to generate         |
| `temperature`  | float  | No       | 0.7     | Sampling temperature (creativity)  |
| `top_p`        | float  | No       | 0.9     | Nucleus sampling threshold         |

### Expected Response (Success)

```json
{
  "status": "success",
  "pertanyaan": "Apa itu Peraturan Desa?",
  "jawaban": "Peraturan Desa adalah peraturan perundang-undangan yang ditetapkan oleh Kepala Desa setelah mendapat persetujuan bersama Badan Permusyawaratan Desa...",
  "model_type": "raft"
}
```

### Expected Response (No Model Loaded)

```json
{
  "status": "error",
  "message": "No model loaded. Use POST /api/load-model first."
}
```
HTTP Status: `503`

### Expected Response (Missing Question)

```json
{
  "status": "error",
  "message": "Field 'pertanyaan' is required and must be non-empty"
}
```
HTTP Status: `400`

---

## 6. POST /api/chat-rag — RAG/RAFT Chat (With Documents)

Send a question along with document context. The model will analyze the documents and generate an answer grounded in the provided context, using Chain-of-Thought (CoT) reasoning.

### Request

```bash
curl -X POST http://localhost:6000/api/chat-rag \
  -H "Content-Type: application/json" \
  -d '{
    "pertanyaan": "Bagaimana prosedur pembentukan Peraturan Desa?",
    "dokumen": [
      "Peraturan Desa ditetapkan oleh Kepala Desa setelah mendapat persetujuan bersama Badan Permusyawaratan Desa. Proses penyusunan Peraturan Desa dimulai dari perencanaan, penyusunan, pembahasan, penetapan, dan pengundangan.",
      "Kepala Desa bersama BPD menyusun Rancangan Peraturan Desa yang dapat berasal dari usulan Kepala Desa atau usulan BPD. Rancangan tersebut dibahas dalam musyawarah desa.",
      "Setelah disetujui bersama, Rancangan Peraturan Desa ditetapkan oleh Kepala Desa dengan membubuhkan tanda tangan dan dicatat dalam Lembaran Desa."
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### Request Parameters

| Field          | Type     | Required | Default | Description                              |
|----------------|----------|----------|---------|------------------------------------------|
| `pertanyaan`   | string   | Yes      | —       | The question to ask                      |
| `dokumen`      | string[] | Yes      | —       | List of document strings as context      |
| `max_tokens`   | int      | No       | 512     | Maximum tokens to generate               |
| `temperature`  | float    | No       | 0.7     | Sampling temperature                     |
| `top_p`        | float    | No       | 0.9     | Nucleus sampling threshold               |

### Expected Response (Success)

```json
{
  "status": "success",
  "pertanyaan": "Bagaimana prosedur pembentukan Peraturan Desa?",
  "analisis": "Dari dokumen yang diberikan, prosedur pembentukan Perdes meliputi beberapa tahapan. Dokumen 1 menjelaskan tahapan umum dari perencanaan hingga pengundangan. Dokumen 2 menjelaskan sumber usulan dan pembahasan. Dokumen 3 menjelaskan tahap akhir penetapan.",
  "jawaban": "Prosedur pembentukan Peraturan Desa meliputi tahapan: (1) Perencanaan, (2) Penyusunan rancangan oleh Kepala Desa bersama BPD, (3) Pembahasan dalam musyawarah desa, (4) Persetujuan bersama, (5) Penetapan oleh Kepala Desa, dan (6) Pengundangan dalam Lembaran Desa.",
  "raw_response": "<CoT>Dari dokumen yang diberikan...</CoT>\nProsedur pembentukan...",
  "model_type": "raft",
  "num_documents": 3
}
```

### Response Fields

| Field           | Description                                                    |
|-----------------|----------------------------------------------------------------|
| `analisis`      | Chain-of-Thought analysis from `<CoT>...</CoT>` block         |
| `jawaban`       | Cleaned final answer (document references removed)             |
| `raw_response`  | Full raw model output before parsing                           |
| `num_documents` | Number of documents that were provided                         |

### Expected Response (Missing Documents)

```json
{
  "status": "error",
  "message": "Field 'dokumen' is required and must be a non-empty list"
}
```
HTTP Status: `400`

---

## Complete Test Workflow

Here is a recommended sequence to test all endpoints in order:

```bash
# Step 1: Check server health
curl -X GET http://localhost:6000/api/health

# Step 2: List available models
curl -X GET http://localhost:6000/api/models

# Step 3: Load the RAFT model
curl -X POST http://localhost:6000/api/load-model \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/workspace/model/raft_merged"}'

# Step 4: Verify model is loaded
curl -X GET http://localhost:6000/api/model-info

# Step 5: Test plain chat
curl -X POST http://localhost:6000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"pertanyaan": "Apa itu Peraturan Desa?"}'

# Step 6: Test RAG chat with documents
curl -X POST http://localhost:6000/api/chat-rag \
  -H "Content-Type: application/json" \
  -d '{
    "pertanyaan": "Bagaimana prosedur pembentukan Peraturan Desa?",
    "dokumen": [
      "Peraturan Desa ditetapkan oleh Kepala Desa setelah mendapat persetujuan bersama BPD.",
      "Rancangan Perdes dapat berasal dari usulan Kepala Desa atau BPD.",
      "Setelah disetujui, Rancangan Perdes ditetapkan dan dicatat dalam Lembaran Desa."
    ]
  }'
```

---

## Error Handling Summary

| HTTP Status | Meaning                                      |
|-------------|----------------------------------------------|
| `200`       | Success                                      |
| `400`       | Bad request (missing/invalid fields)         |
| `500`       | Internal server error (model failure, etc.)  |
| `503`       | Service unavailable (no model loaded)        |

All error responses follow the format:
```json
{
  "status": "error",
  "message": "Description of the error"
}
```
