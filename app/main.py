import os, io, json, tempfile, time, hashlib, csv, datetime
from collections import defaultdict, deque
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
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
CACHE_DIR = "cache"
RESULTS_CSV = os.path.join(UPLOAD_DIR, "results.csv")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Simple in-memory rate limit (per instance)
RATE_LIMIT = 10       # requests
RATE_WINDOW_SEC = 3600
ip_hits: dict[str, deque] = defaultdict(deque)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

def _rate_limit_ok(ip: str) -> bool:
    now = time.time()
    dq = ip_hits[ip]
    # purge old
    while dq and now - dq[0] > RATE_WINDOW_SEC:
        dq.popleft()
    if len(dq) >= RATE_LIMIT:
        return False
    dq.append(now)
    return True

def _sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def _load_cache(key: str):
    p = _cache_path(key)
    if os.path.isfile(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _save_cache(key: str, value: dict):
    p = _cache_path(key)
    try:
        with open(p, "w") as f:
            json.dump(value, f, indent=2)
    except Exception:
        pass

def _append_csv(row: dict):
    exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp", "filename", "duration_sec", "loudness_dbfs",
            "title", "artist", "timecode", "spotify_url", "apple_url",
            "cache_hit", "processing_time_sec"
        ])
        if not exists:
            w.writeheader()
        w.writerow(row)

def _read_last_csv(n: int = 10):
    if not os.path.isfile(RESULTS_CSV):
        return []
    try:
        with open(RESULTS_CSV, "r") as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    recent = _read_last_csv(10)
    return templates.TemplateResponse("index.html", {"request": request, "recent": recent})

@app.get("/results.csv")
def download_csv():
    if not os.path.isfile(RESULTS_CSV):
        return PlainTextResponse("No results yet.", status_code=404)
    return FileResponse(RESULTS_CSV, media_type="text/csv", filename="results.csv")

@app.post("/analyze", response_class=HTMLResponse)
def analyze(request: Request, file: UploadFile = File(...), start_seconds: int = Form(30)):
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limit_ok(client_ip):
        raise HTTPException(429, f"Liiga palju päringuid. Proovi uuesti hiljem. (limiit {RATE_LIMIT}/h)")

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

    # Decode via pydub
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

    # Clip parameters
    clip_len_ms = 15_000
    start_ms_default = max(0, min(int(start_seconds * 1000), max(0, len(seg) - clip_len_ms)))
    start_ms = start_ms_default
    clip = seg[start_ms:start_ms + clip_len_ms]

    # Caching key: audio sha1 + start_ms + clip_len
    cache_key = f"{_sha1(raw)}_{start_ms}_{clip_len_ms}"
    cached = _load_cache(cache_key)

    audd_json = None
    cache_hit = False
    if AUDD_API_TOKEN:
        if cached:
            audd_json = cached
            cache_hit = True
        else:
            try:
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
                _save_cache(cache_key, audd_json)
            except requests.Timeout:
                audd_json = {"error": "AudD timeout (30s). Proovi uuesti või nihuta alguspunkti."}
            except Exception as e:
                audd_json = {"error": str(e)}
    else:
        audd_json = {"warning": "AUDD_API_TOKEN puudub; lisa see Renderi Environmentis."}

    # Parse AudD for UI
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

    # Log to CSV
    _append_csv({
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "filename": filename,
        "duration_sec": duration_sec,
        "loudness_dbfs": dbfs,
        "title": (audd_summary or {}).get("title") if audd_summary else "",
        "artist": (audd_summary or {}).get("artist") if audd_summary else "",
        "timecode": (audd_summary or {}).get("timecode") if audd_summary else "",
        "spotify_url": spotify_url or "",
        "apple_url": apple_url or "",
        "cache_hit": str(cache_hit),
        "processing_time_sec": round(t1 - t0, 3),
    })

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "filename": filename,
            "duration": duration_sec,
            "loudness": dbfs,
            "json_file": f"/uploads/{filename}_analysis.json",  # deprecated; keep for compatibility
            "audd_result": audd_json,
            "audd_summary": audd_summary,
            "audd_cover": cover_url,
            "spotify_url": spotify_url,
            "apple_url": apple_url,
            "time_total": round(t1 - t0, 3),
            "start_seconds": int(start_seconds),
            "cache_hit": cache_hit,
            "recent": _read_last_csv(10),
        },
    )
