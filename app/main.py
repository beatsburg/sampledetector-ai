import os, io, json, tempfile, time
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
import imageio_ffmpeg
import requests

# Use bundled ffmpeg binary (no apt-get needed)
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN")
MAX_FILE_SIZE_MB = 20
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
def analyze(request: Request, file: UploadFile = File(...)):
    t0 = time.perf_counter()
    if not file:
        raise HTTPException(400, "Fail on nõutud")
    raw = file.file.read()
    size_mb = len(raw) / (1024*1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"Fail on {size_mb:.1f} MB; limiit {MAX_FILE_SIZE_MB} MB.")

    filename = file.filename or "audio"
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(raw)

    # Decode via pydub (supports mp3/m4a/wav/ogg/etc. via imageio-ffmpeg)
    try:
        ext = os.path.splitext(filename)[1].lower() or ".bin"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        seg = AudioSegment.from_file(tmp_path)
    except Exception as e:
        raise HTTPException(415, f"Audio decode failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    duration_sec = round(len(seg) / 1000.0, 2)
    dbfs = round(seg.dBFS, 2) if seg.dBFS != float("-inf") else -60.0

    # --- AudD lookup: send only a 15s clip (faster & cheaper) ---
    audd_json = None
    if AUDD_API_TOKEN:
        try:
            start_ms_default = 30_000
            clip_len_ms = 15_000
            start_ms = min(start_ms_default, max(0, len(seg) - clip_len_ms))
            clip = seg[start_ms:start_ms + clip_len_ms]

            buf = io.BytesIO()
            clip.export(buf, format="mp3")
            buf.seek(0)

            resp = requests.post(
                "https://api.audd.io/",
                data={
                    "api_token": AUDD_API_TOKEN,
                    "return": "timecode,apple_music,spotify",
                },
                files={"file": ("clip.mp3", buf, "audio/mpeg")},
                timeout=30
            )
            audd_json = resp.json()
        except Exception as e:
            audd_json = {"error": str(e)}

    # Parse AudD result for UI
    audd_summary = None
    cover_url = spotify_url = apple_url = None
    if isinstance(audd_json, dict) and audd_json.get("result"):
        r = audd_json.get("result") or {}
        audd_summary = {
            "title": r.get("title"),
            "artist": r.get("artist"),
            "album": r.get("album"),
            "timecode": r.get("timecode"),
            "label": r.get("label"),
            "release_date": r.get("release_date"),
            "song_link": r.get("song_link"),
        }
        apple = r.get("apple_music") or {}
        sp = r.get("spotify") or {}
        cover_url = (apple.get("artwork") or {}).get("url")
        spotify_url = (sp.get("external_urls") or {}).get("spotify")
        apple_url = apple.get("url")

    t1 = time.perf_counter()
    result = {
        "filename": filename,
        "filesize_mb": round(size_mb, 2),
        "duration_sec": duration_sec,
        "loudness_dbfs": dbfs,
        "processing_time_sec": round(t1 - t0, 3),
        "audd": audd_json
    }

    json_path = os.path.join(UPLOAD_DIR, f"{filename}_analysis.json")
    with open(json_path, "w") as jf:
        json.dump(result, jf, indent=2)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "filename": filename,
            "duration": duration_sec,
            "loudness": dbfs,
            "json_file": f"/uploads/{os.path.basename(json_path)}",
            "audd_result": audd_json,
            "audd_summary": audd_summary,
            "audd_cover": cover_url,
            "spotify_url": spotify_url,
            "apple_url": apple_url,
            "time_total": result["processing_time_sec"],
        },
    )

@app.get("/uploads/{filename}")
def get_uploaded(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)
