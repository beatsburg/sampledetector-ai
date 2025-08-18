# SampleDetector v1.3.4 (stable)

- Multi-file upload + results table
- Per-file JSON saved to /uploads/*_analysis.json (+ link in UI)
- /analyze.json endpoint (API)
- Lightweight deps; Render Free friendly

Deploy:
1) Commit & Push
2) Render → Manual Deploy → Clear build cache → Deploy latest commit
3) Env: AUDD_API_TOKEN, ADMIN_TOKEN (optional), PYTHON_VERSION=3.11.9
