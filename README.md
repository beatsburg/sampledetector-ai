# SampleDetector v2.0-alpha – Sample Origin

Endpoints:
- `POST /index-samples` (form-data: zipfile_upload=ZIP)
- `POST /sample-origin` (HTML result; form-data: track=audio, start_seconds=0)
- `POST /sample-origin.json` (JSON result)
- `GET /index/meta`, `GET /index/clear`, `GET /health`

Params & limits:
- Reference: max 200 files, 120 s/each; Windows: 2 s with 0.5 s hop
- Query: max 120 s; pitch: -4..+4; tempo: 0.9/1.0/1.1; top-5 NNs per window
- Similarity threshold: 0.80 (alpha)

Runtime: Python 3.11.9; pinned numpy/scipy/librosa; ANN via Annoy.
