import os, time, json
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
import aiohttp
from aiohttp import ClientTimeout

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_FILE_SIZE_MB = 20
MAX_ANALYZE_SECONDS = 90.0
TARGET_SR = 22050
RES_TYPE = "kaiser_fast"

AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN")

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
        raise HTTPException(status_code=400, detail=f"Fail on {size_mb:.1f} MB; limiit {MAX_FILE_SIZE_MB} MB. Proovi lühemat või tihendatud MP3/WAV-i.")

    with open(file_path, "wb") as f:
        f.write(raw)
    t_saved = time.perf_counter()

    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True, duration=MAX_ANALYZE_SECONDS, res_type=RES_TYPE)
    duration = librosa.get_duration(y=y, sr=sr)
    t_loaded = time.perf_counter()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate='median'))
    t_tempo = time.perf_counter()

    waveform_path = os.path.join(UPLOAD_FOLDER, f"{filename}_waveform.png")
    _save_waveform_png(y, sr, waveform_path)
    t_wave = time.perf_counter()

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

    audd_result = None
    if AUDD_API_TOKEN:
        try:
            timeout = ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                form = aiohttp.FormData()
                form.add_field("api_token", AUDD_API_TOKEN)
                form.add_field("return", "timecode,apple_music,spotify")
                form.add_field("file", open(file_path, "rb"), filename=filename)
                async with session.post("https://api.audd.io/", data=form) as resp:
                    audd_result = await resp.json()
        except Exception as e:
            audd_result = {"error": str(e)}
    t_audd = time.perf_counter()

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
            "audd": round(t_audd - t_pdf, 3),
            "total": round(t_audd - t0, 3)
        }
    }
    json_path = os.path.join(UPLOAD_FOLDER, f"{filename}_analysis.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

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
        "audd_result": audd_result,
        "timing": json_data["timing_sec"]
    })

@app.get("/uploads/{filename}")
async def get_file(filename: str):
    return FileResponse(path=os.path.join(UPLOAD_FOLDER, filename))
