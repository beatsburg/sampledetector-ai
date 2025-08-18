import os, io, json, tempfile, time, csv, datetime, threading
from collections import defaultdict, deque
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import imageio_ffmpeg
import requests
import hashlib

VERSION = "1.3.4-stable"
START_TIME = time.time()

# ffmpeg path for pydub
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# --- Config via env ---
AUDD_API_TOKEN = os.getenv("AUDD_API_TOKEN")            # optional
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")                  # protects admin/CSV/clear-cache
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")         # CORS
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "86400"))   # 24h
RESULTS_CSV = os.path.join(UPLOAD_DIR, "results.csv")
AUTO_CLEAN_MIN = int(os.getenv("AUTO_CLEAN_MIN", "60"))     # remove uploads older than X min
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "30"))             # requests per window (raised for batch)
RATE_WINDOW_SEC = int(os.getenv("RATE_WINDOW_SEC", "3600")) # seconds

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --- App & middleware ---
app = FastAPI(title="SampleDetector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory="app/templates")

# --- Helpers ---
ip_hits: dict[str, deque] = defaultdict(deque)

def _rate_limit_ok(ip: str, hits: int = 1) -> bool:
    now = time.time()
    dq = ip_hits[ip]
    while dq and now - dq[0] > RATE_WINDOW_SEC:
        dq.popleft()
    if len(dq) + hits > RATE_LIMIT:
        return False
    for _ in range(hits):
        dq.append(now)
    return True

def _sha1(data: bytes) -> str:
    h = hashlib.sha1(); h.update(data); return h.hexdigest()

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def _load_cache(key: str):
    p = _cache_path(key)
    if os.path.isfile(p):
        try:
            with open(p, "r") as f:
                data = json.load(f)
            if time.time() - data.get("_ts", 0) <= CACHE_TTL_SEC:
                return data
        except Exception:
            return None
    return None

def _save_cache(key: str, value: dict):
    p = _cache_path(key)
    try:
        value = dict(value or {}); value["_ts"] = time.time()
        with open(p, "w") as f:
            json.dump(value, f, indent=2)
    except Exception:
        pass

def _append_csv(row: dict):
    exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=[
            "timestamp", "filename", "duration_sec", "loudness_dbfs",
            "title", "artist", "timecode", "spotify_url", "apple_url",
            "cache_hit", "processing_time_sec"
        ])
        if not exists: w.writeheader()
        w.writerow(row)

def _read_last_csv(n: int = 20):
    if not os.path.isfile(RESULTS_CSV): return []
    try:
        with open(RESULTS_CSV, "r") as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []

def require_admin(request: Request):
    token = request.headers.get("x-admin-token") or request.query_params.get("admin")
    if not ADMIN_TOKEN:
        raise HTTPException(403, "ADMIN_TOKEN pole seadistatud")
    if token != ADMIN_TOKEN:
        raise HTTPException(401, "Vale või puudu x-admin-token")
    return True

def _auto_clean_loop():
    while True:
        try:
            cutoff = time.time() - (AUTO_CLEAN_MIN * 60)
            for root, _, files in os.walk(UPLOAD_DIR):
                for nm in files:
                    p = os.path.join(root, nm)
                    try:
                        if os.path.getmtime(p) < cutoff:
                            os.remove(p)
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(60)

threading.Thread(target=_auto_clean_loop, daemon=True).start()

# --- Health ---
@app.get("/health")
def health():
    return {"ok": True, "version": VERSION, "uptime_sec": round(time.time() - START_TIME, 1)}

# --- Home ---
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    recent = _read_last_csv(20)
    return templates.TemplateResponse("index.html", {"request": request, "recent": recent, "version": VERSION})

# --- CSV & cache ---
@app.get("/results.csv")
def download_csv(request: Request, ok: bool = Depends(require_admin)):
    if not os.path.isfile(RESULTS_CSV):
        return PlainTextResponse("No results yet.", status_code=404)
    return FileResponse(RESULTS_CSV, media_type="text/csv", filename="results.csv")

@app.post("/clear-cache")
def clear_cache(request: Request, ok: bool = Depends(require_admin)):
    n = 0
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".json"):
            try:
                os.remove(os.path.join(CACHE_DIR, f)); n += 1
            except Exception:
                pass
    return {"cleared": n}

# --- Core analyze helper ---
def _analyze_one(raw: bytes, filename: str, start_seconds: int, clip_seconds: int):
    size_mb = len(raw) / (1024*1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"{filename}: {size_mb:.1f} MB; limiit {MAX_FILE_SIZE_MB} MB.")

    # Save original
    safe_name = (filename or 'audio').replace('/', '_').replace('\\', '_')[:200]
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as f:
        f.write(raw)

    # Decode using pydub
    try:
        ext = os.path.splitext(safe_name)[1].lower() or ".bin"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw); tmp_path = tmp.name
        seg = AudioSegment.from_file(tmp_path)
    except Exception as e:
        raise HTTPException(415, f"{safe_name}: decode failed: {e}")
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

    duration_sec = round(len(seg) / 1000.0, 2)
    dbfs = round(seg.dBFS, 2) if seg.dBFS != float("-inf") else -60.0

    # Clip bounds
    clip_len_ms = max(5, min(clip_seconds, 25)) * 1000
    start_ms_default = max(0, min(int(start_seconds * 1000), max(0, len(seg) - clip_len_ms)))
    clip = seg[start_ms_default:start_ms_default + clip_len_ms]

    # Cache key
    key = f"{_sha1(raw)}_{start_ms_default}_{clip_len_ms}"
    cached = _load_cache(key)

    audd_json = None
    cache_hit = False
    if AUDD_API_TOKEN:
        if cached:
            audd_json = cached; cache_hit = True
        else:
            try:
                buf = io.BytesIO(); clip.export(buf, format="mp3"); buf.seek(0)
                resp = requests.post(
                    "https://api.audd.io/",
                    data={"api_token": AUDD_API_TOKEN, "return": "timecode,apple_music,spotify"},
                    files={"file": ("clip.mp3", buf, "audio/mpeg")},
                    timeout=30
                )
                audd_json = resp.json(); _save_cache(key, audd_json)
            except requests.Timeout:
                audd_json = {"error": "AudD timeout (30s). Proovi uuesti või nihuta alguspunkti."}
            except Exception as e:
                audd_json = {"error": str(e)}
    else:
        audd_json = {"warning": "AUDD_API_TOKEN puudub; lisa see Renderi Environmentis."}

    # Summary
    summary = None; cover_url = spotify_url = apple_url = None
    if isinstance(audd_json, dict) and audd_json.get("result"):
        r = audd_json.get("result") or {}
        summary = {"title": r.get("title"), "artist": r.get("artist"), "album": r.get("album"),
                   "timecode": r.get("timecode"), "label": r.get("label"), "release_date": r.get("release_date"),
                   "song_link": r.get("song_link")}
        apple = r.get("apple_music") or {}; sp = r.get("spotify") or {}
        cover_url = (apple.get("artwork") or {}).get("url")
        spotify_url = (sp.get("external_urls") or {}).get("spotify")
        apple_url = apple.get("url")

    result = {
        "filename": safe_name,
        "duration_sec": duration_sec,
        "loudness_dbfs": dbfs,
        "start_seconds": start_ms_default // 1000,
        "clip_seconds": clip_len_ms // 1000,
        "audd": audd_json,
        "audd_summary": summary,
        "spotify_url": spotify_url,
        "apple_url": apple_url,
        "cover_url": cover_url,
        "cache_hit": cache_hit,
    }

    # Save per-file JSON for UI link
    json_path = os.path.join(UPLOAD_DIR, f"{safe_name}_analysis.json")
    with open(json_path, "w") as jf: json.dump(result, jf, indent=2)

    # CSV log
    _append_csv({
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "filename": safe_name,
        "duration_sec": duration_sec,
        "loudness_dbfs": dbfs,
        "title": (summary or {}).get("title") if summary else "",
        "artist": (summary or {}).get("artist") if summary else "",
        "timecode": (summary or {}).get("timecode") if summary else "",
        "spotify_url": spotify_url or "",
        "apple_url": apple_url or "",
        "cache_hit": str(cache_hit),
        "processing_time_sec": "",  # optional
    })

    return result, json_path

# --- Analyze routes ---
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_html(
    request: Request,
    files: List[UploadFile] = File(...),
    start_seconds: int = Form(30),
    clip_seconds: int = Form(15),
    format: Optional[str] = Form(None)
):
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limit_ok(client_ip, hits=len(files)):
        raise HTTPException(429, f"Liiga palju päringuid. Limiit {RATE_LIMIT}/h.")

    t0 = time.perf_counter()
    results, errors = [], []

    for f in files:
        try:
            raw = await f.read()
            res, json_path = _analyze_one(raw, f.filename or "audio", start_seconds, clip_seconds)
            res["json_url"] = f"/uploads/{os.path.basename(json_path)}"
            results.append(res)
        except Exception as e:
            errors.append(str(e))

    total_time = round(time.perf_counter() - t0, 3)
    wants_json = (format == "json") or ("application/json" in (request.headers.get("accept") or ""))
    payload = {"version": VERSION, "count": len(results), "errors": errors, "results": results, "processing_time_sec": total_time}
    if wants_json:
        return JSONResponse(payload)

    recent = _read_last_csv(20)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "version": VERSION,
        "batch": True,
        "results": results,
        "errors": errors,
        "recent": recent,
        "total_time": total_time
    })

@app.post("/analyze.json")
async def analyze_json(
    request: Request,
    files: List[UploadFile] = File(...),
    start_seconds: int = Form(30),
    clip_seconds: int = Form(15)
):
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limit_ok(client_ip, hits=len(files)):
        raise HTTPException(429, f"Liiga palju päringuid. Limiit {RATE_LIMIT}/h.")

    t0 = time.perf_counter()
    results, errors = [], []
    for f in files:
        try:
            raw = await f.read()
            res, json_path = _analyze_one(raw, f.filename or "audio", start_seconds, clip_seconds)
            res["json_url"] = f"/uploads/{os.path.basename(json_path)}"
            results.append(res)
        except Exception as e:
            errors.append(str(e))

    return {"version": VERSION, "count": len(results), "errors": errors, "results": results, "processing_time_sec": round(time.perf_counter()-t0, 3)}

# --- Admin bare page (optional) ---
@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, ok: bool = Depends(require_admin)):
    return templates.TemplateResponse("admin.html", {"request": request, "version": VERSION})
