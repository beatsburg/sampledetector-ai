import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from reportlab.pdfgen import canvas
from datetime import datetime
import json
import aiohttp

UPLOAD_FOLDER = "uploads"
MAX_FILE_SIZE_MB = 10  # max allowed file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_audio(request: Request, file: UploadFile = File(...)):
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Kontrolli faili suurust (max 10 MB)
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"Fail on liiga suur ({size_mb:.1f} MB). Maksimum on {MAX_FILE_SIZE_MB} MB.")

    # Salvesta fail
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        # Töötlus
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Waveform
        waveform_path = os.path.join(UPLOAD_FOLDER, f"{filename}_waveform.png")
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title("Waveform")
        plt.savefig(waveform_path)
        plt.close('all')

        # PDF
        pdf_path = os.path.join(UPLOAD_FOLDER, f"{filename}_report.pdf")
        c = canvas.Canvas(pdf_path)
        c.setFont("Helvetica", 14)
        c.drawString(50, 800, f"SampleGuard Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 780, f"Filename: {filename}")
        c.drawString(50, 760, f"Duration: {duration:.2f} seconds")
        c.drawString(50, 740, f"Tempo (BPM): {tempo:.2f}")
        c.drawString(50, 720, f"Generated: {datetime.utcnow().isoformat()} UTC")
        c.drawImage(waveform_path, 50, 400, width=500, height=250)
        c.save()

        # JSON
        json_data = {
            "filename": filename,
            "duration_sec": duration,
            "tempo_bpm": tempo
        }
        json_path = os.path.join(UPLOAD_FOLDER, f"{filename}_analysis.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # AudD API
        audd_result = None
        if AUDD_API_TOKEN:
            async with aiohttp.ClientSession() as session:
                with open(file_path, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field("api_token", AUDD_API_TOKEN)
                    data.add_field("file", f, filename=filename)
                    async with session.post("https://api.audd.io/", data=data) as resp:
                        audd_result = await resp.json()

        return templates.TemplateResponse("index.html", {
            "request": request,
            "filename": filename,
            "duration": f"{duration:.2f}",
            "tempo": f"{tempo:.2f}",
            "pdf_report": f"/uploads/{filename}_report.pdf",
            "waveform_image": f"/uploads/{filename}_waveform.png",
            "json_file": f"/uploads/{filename}_analysis.json",
            "audd_result": audd_result
        })

    finally:
        del y, sr
        plt.close('all')

@app.get("/uploads/{filename}")
async def get_file(filename: str):
    return FileResponse(path=os.path.join(UPLOAD_FOLDER, filename))
