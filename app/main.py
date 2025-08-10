import os, io, time, json, tempfile, traceback
import numpy as np
import soundfile as sf
import librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from reportlab.pdfgen import canvas
from datetime import datetime

from pydub import AudioSegment
import imageio_ffmpeg
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_FILE_SIZE_MB = 20
MAX_ANALYZE_SECONDS = 60.0   # slightly lower to reduce RAM
TARGET_SR = 22050
RES_TYPE = "scipy"           # avoid resampy requirement

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def _save_waveform_png(y, sr, out_path):
    try:
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
    finally:
        plt.close('all')

def _safe_load_audio(file_bytes: bytes, filename: str):
    ext = os.path.splitext(filename.lower())[1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True, duration=MAX_ANALYZE_SECONDS, res_type=RES_TYPE)
        return y, sr
    except Exception:
        try:
            audio = AudioSegment.from_file(tmp_path)
            if len(audio) > MAX_ANALYZE_SECONDS * 1000:
                audio = audio[:int(MAX_ANALYZE_SECONDS * 1000)]
            buf = io.BytesIO()
            audio.export(buf, format="wav"); buf.seek(0)
            y, sr = librosa.load(buf, sr=TARGET_SR, mono=True, res_type=RES_TYPE)
            return y, sr
        except Exception as e:
            raise HTTPException(status_code=415, detail=f"Audio decode failed: {e}")
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
    try:
        t0 = time.perf_counter()
        filename = file.filename or "audio"
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        raw = await file.read()
        size_mb = len(raw) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"Fail on {size_mb:.1f} MB; limiit {MAX_FILE_SIZE_MB} MB.")

        with open(file_path, "wb") as f:
            f.write(raw)

        y, sr = _safe_load_audio(raw, filename)
        duration = librosa.get_duration(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate='median'))

        waveform_path = os.path.join(UPLOAD_FOLDER, f"{filename}_waveform.png")
        _save_waveform_png(y, sr, waveform_path)

        pdf_path = os.path.join(UPLOAD_FOLDER, f"{filename}_report.pdf")
        c = canvas.Canvas(pdf_path)
        c.setFont("Helvetica", 14); c.drawString(50, 800, "SampleDetector Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 780, f"Filename: {filename}")
        c.drawString(50, 760, f"Analyzed duration: {duration:.2f} s (cap {MAX_ANALYZE_SECONDS:.0f}s)")
        c.drawString(50, 740, f"Tempo (BPM): {tempo:.1f}")
        c.setFont("Helvetica", 9); c.drawString(50, 710, f"Resampler: {RES_TYPE}, SR: {TARGET_SR}")
        try:
            c.drawImage(waveform_path, 50, 440, width=500, height=200)
        except Exception:
            pass
        c.save()

        json_data = {
            "filename": filename,
            "filesize_mb": round(size_mb, 2),
            "analyzed_seconds": round(duration, 2),
            "tempo_bpm": round(tempo, 1),
        }
        json_path = os.path.join(UPLOAD_FOLDER, f"{filename}_analysis.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        del y, onset_env
        return templates.TemplateResponse("index.html", {
            "request": request,
            "filename": filename,
            "duration": f"{duration:.2f}",
            "tempo": f"{tempo:.1f}",
            "pdf_report": f"/uploads/{os.path.basename(pdf_path)}",
            "waveform_image": f"/uploads/{os.path.basename(waveform_path)}",
            "json_file": f"/uploads/{os.path.basename(json_path)}"
        })

    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    except Exception as e:
        # Surface full traceback as JSON to avoid opaque 500
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"detail": str(e), "traceback": tb})

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/uploads/{filename}")
async def get_file(filename: str):
    return FileResponse(path=os.path.join(UPLOAD_FOLDER, filename))
