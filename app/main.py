import os, io, time, json, tempfile
import numpy as np
import soundfile as sf
import librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from reportlab.pdfgen import canvas
from datetime import datetime

# FFmpeg-less fallback using imageio-ffmpeg + pydub
from pydub import AudioSegment
import imageio_ffmpeg

# Configure pydub to use bundled ffmpeg binary from imageio-ffmpeg
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_FILE_SIZE_MB = 20
MAX_ANALYZE_SECONDS = 90.0
TARGET_SR = 22050
RES_TYPE = "kaiser_fast"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def _save_waveform_png(y, sr, out_path):
    if len(y) > 2000:
        step = len(y) // 2000
        y_plot = y[::step]
        sr_plot = sr // max(1, len(y)//len(y_plot))
    else:
        y_plot = y
        sr_plot = sr
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y_plot, sr=sr_plot)
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close('all')

def _safe_load_audio(file_bytes: bytes, filename: str):
    # Try fast path with librosa+soundfile (works for WAV/OGG/FLAC)
    ext = os.path.splitext(filename.lower())[1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        # For WAV/OGG etc this is efficient
        y, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True, duration=MAX_ANALYZE_SECONDS, res_type=RES_TYPE)
        return y, sr
    except Exception:
        # Fallback: decode with pydub (uses imageio-ffmpeg binary) then hand to librosa
        try:
            audio = AudioSegment.from_file(tmp_path)
            # Truncate to MAX_ANALYZE_SECONDS to reduce RAM
            if len(audio) > MAX_ANALYZE_SECONDS * 1000:
                audio = audio[:int(MAX_ANALYZE_SECONDS * 1000)]
            # Export to WAV bytes in-memory
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            buf.seek(0)
            y, sr = librosa.load(buf, sr=TARGET_SR, mono=True, res_type=RES_TYPE)
            return y, sr
        except Exception as e:
            raise HTTPException(status_code=415, detail=f"Audio decode failed. Try WAV/OGG või teistsugune MP3/M4A. ({e})")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_audio(request: Request, file: UploadFile = File(...)):
    t0 = time.perf_counter()
    filename = file.filename or "audio"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"Fail on {size_mb:.1f} MB; limiit {MAX_FILE_SIZE_MB} MB.")

    # Persist original upload
    with open(file_path, "wb") as f:
        f.write(raw)
    t_saved = time.perf_counter()

    # Decode with safe loader (no system ffmpeg required)
    y, sr = _safe_load_audio(raw, filename)
    duration = librosa.get_duration(y=y, sr=sr)
    t_loaded = time.perf_counter()

    # Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate='median'))
    t_tempo = time.perf_counter()

    # Waveform
    waveform_path = os.path.join(UPLOAD_FOLDER, f"{filename}_waveform.png")
    _save_waveform_png(y, sr, waveform_path)
    t_wave = time.perf_counter()

    # PDF
    pdf_path = os.path.join(UPLOAD_FOLDER, f"{filename}_report.pdf")
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 14); c.drawString(50, 800, "SampleDetector Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 780, f"Filename: {filename}")
    c.drawString(50, 760, f"Analyzed duration: {duration:.2f} s (cap {MAX_ANALYZE_SECONDS:.0f}s)")
    c.drawString(50, 740, f"Tempo (BPM): {tempo:.1f}")
    c.drawString(50, 720, f"Generated: {datetime.utcnow().isoformat()}Z")
    try:
        c.drawImage(waveform_path, 50, 440, width=500, height=200)
    except Exception:
        pass
    c.save()
    t_pdf = time.perf_counter()

    json_data = {
        "filename": filename,
        "filesize_mb": round(size_mb, 2),
        "analyzed_seconds": round(duration, 2),
        "tempo_bpm": round(tempo, 1),
        "timing_sec": {
            "save": round(t_saved - t0, 3),
            "load": round(t_loaded - t_saved, 3),
            "tempo": round(t_tempo - t_loaded, 3),
            "waveform": round(t_wave - t_tempo, 3),
            "pdf": round(t_pdf - t_wave, 3),
            "total": round(t_pdf - t0, 3)
        }
    }
    json_path = os.path.join(UPLOAD_FOLDER, f"{filename}_analysis.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # cleanup
    del y, onset_env
    plt.close('all')

    return templates.TemplateResponse("index.html", {
        "request": request,
        "filename": filename,
        "duration": f"{duration:.2f}",
        "tempo": f"{tempo:.1f}",
        "pdf_report": f"/uploads/{os.path.basename(pdf_path)}",
        "waveform_image": f"/uploads/{os.path.basename(waveform_path)}",
        "json_file": f"/uploads/{os.path.basename(json_path)}",
        "timing": json_data["timing_sec"]
    })

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/uploads/{filename}")
async def get_file(filename: str):
    return FileResponse(path=os.path.join(UPLOAD_FOLDER, filename))
