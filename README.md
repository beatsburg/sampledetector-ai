# sampledetector-ai (minimal, AUDD ready)

- Python: 3.11.9 (`runtime.txt`)
- No SciPy/Librosa. Uses **pydub + imageio-ffmpeg** for decoding, and **requests** for AudD.
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## ENV
- `AUDD_API_TOKEN` (optional): add in Render → Settings → Environment

## Deploy on Render
1. Commit & push to GitHub
2. Build command: `pip install --no-cache-dir -r requirements.txt`
3. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Manual Deploy → Clear build cache → Deploy latest commit
