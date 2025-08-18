import os, io, json, tempfile, time, csv, datetime, threading, math
from collections import defaultdict, Counter, deque
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import imageio_ffmpeg
import requests
import hashlib
import numpy as np

VERSION = "1.4.0-auto-catalog"
START_TIME = time.time()

# ffmpeg path for pydub
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# --- Config via env ---
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CATALOG_DIR = os.getenv("CATALOG_DIR", "catalog")
INDEX_DIR = os.getenv("INDEX_DIR", "index")
ITUNES_LIMIT = int(os.getenv("ITUNES_LIMIT", "25"))  # per query
CATALOG_MAX = int(os.getenv("CATALOG_MAX", "150"))   # total cap
SR = int(os.getenv("SR", "22050"))
CLIP_SECONDS = int(os.getenv("CLIP_SECONDS", "30"))  # use first 30s of preview
QUERY_SECONDS = int(os.getenv("QUERY_SECONDS", "90"))# analyze first 90s of user track
PEAKS_PER_FRAME = int(os.getenv("PEAKS_PER_FRAME", "5"))
PAIR_DT_MIN = int(os.getenv("PAIR_DT_MIN", "3"))
PAIR_DT_MAX = int(os.getenv("PAIR_DT_MAX", "35"))
FANOUT = int(os.getenv("FANOUT", "5"))
MATCH_VOTE_THRESHOLD = int(os.getenv("MATCH_VOTE_THRESHOLD", "20"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "30"))
RATE_WINDOW_SEC = int(os.getenv("RATE_WINDOW_SEC", "3600"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CATALOG_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# --- App & middleware ---
app = FastAPI(title="SampleDetector · AutoCatalog")
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

# --- Rate limit ---
ip_hits: dict[str, deque] = defaultdict(deque)
def _rate_limit_ok(ip: str, hits: int = 1) -> bool:
    now = time.time()
    dq = ip_hits[ip]
    while dq and now - dq[0] > RATE_WINDOW_SEC:
        dq.popleft()
    if len(dq) + hits > RATE_LIMIT:
        return False
    for _ in range(hits): dq.append(now)
    return True

def require_admin(request: Request):
    token = request.headers.get("x-admin-token") or request.query_params.get("admin")
    if not ADMIN_TOKEN: raise HTTPException(403, "ADMIN_TOKEN pole seadistatud")
    if token != ADMIN_TOKEN: raise HTTPException(401, "Vale või puudu x-admin-token")
    return True

# --- Fingerprint (constellation-like) ---
def stft_magnitude(y: np.ndarray, sr: int, win: int = 4096, hop: int = 512) -> np.ndarray:
    # Hann window
    w = np.hanning(win)
    n_frames = 1 + (len(y) - win) // hop if len(y) >= win else 1
    spec = np.empty((win//2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = y[start:start+win]
        if len(frame) < win:
            f = np.zeros(win, dtype=np.float32); f[:len(frame)] = frame
            frame = f
        frame = frame * w
        fft = np.fft.rfft(frame)
        mag = np.abs(fft)
        spec[:, i] = mag
    return spec

def find_peaks_per_frame(spec: np.ndarray, n_peaks: int) -> List[List[int]]:
    peaks = []
    for i in range(spec.shape[1]):
        col = spec[:, i]
        if n_peaks >= len(col):
            idx = np.argsort(col)[::-1]
        else:
            idx = np.argpartition(col, -n_peaks)[-n_peaks:]
            idx = idx[np.argsort(col[idx])[::-1]]
        peaks.append(idx.tolist())
    return peaks

def hash_pairs(peaks: List[List[int]], fanout: int, dt_min: int, dt_max: int) -> List[tuple]:
    pairs = []
    for t1, freqs in enumerate(peaks):
        for f1_i, f1 in enumerate(freqs[:fanout]):
            # pair with future frames within [dt_min, dt_max]
            for dt in range(dt_min, dt_max+1):
                t2 = t1 + dt
                if t2 >= len(peaks): break
                for f2 in peaks[t2][:fanout]:
                    # build 32-bit-ish hash from (f1, f2, dt), coarse quantization
                    h = (int(f1) & 0x3FF) | ((int(f2) & 0x3FF) << 10) | ((int(dt) & 0x3FF) << 20)
                    pairs.append((h, t1))
    return pairs

def mono_resample(raw: bytes, ext: str, sr: int, max_seconds: int) -> np.ndarray:
    # decode with pydub (ffmpeg), resample via raw export -> numpy
    seg = AudioSegment.from_file(io.BytesIO(raw), format=ext.replace(".","") if ext else None)
    if len(seg) > max_seconds * 1000:
        seg = seg[:max_seconds * 1000]
    seg = seg.set_channels(1).set_frame_rate(sr).set_sample_width(2)  # 16-bit
    samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples

def fingerprint_from_audio(raw: bytes, filename: str, sr: int, max_seconds: int):
    ext = os.path.splitext(filename)[1].lower()
    y = mono_resample(raw, ext, sr, max_seconds)
    spec = stft_magnitude(y, sr, win=4096, hop=512)
    peaks = find_peaks_per_frame(spec, PEAKS_PER_FRAME)
    pairs = hash_pairs(peaks, FANOUT, PAIR_DT_MIN, PAIR_DT_MAX)
    return pairs  # list of (hash, t1)

# --- Index utils ---
META_JSON = os.path.join(INDEX_DIR, "meta.json")
INDEX_JSONL = os.path.join(INDEX_DIR, "index.jsonl")

def load_meta() -> Dict[str, Any]:
    if os.path.isfile(META_JSON):
        with open(META_JSON, "r") as f: return json.load(f)
    return {"tracks": {}, "count": 0}

def save_meta(meta: Dict[str, Any]):
    with open(META_JSON, "w") as f: json.dump(meta, f, indent=2)

def append_index_rows(rows: List[Dict[str, Any]]):
    with open(INDEX_JSONL, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def iter_index_records():
    if not os.path.isfile(INDEX_JSONL): return
    with open(INDEX_JSONL, "r") as f:
        for line in f:
            try: yield json.loads(line)
            except Exception: continue

# Build in-memory lookup: hash -> list[(track_id, t_ref)]
def build_lookup():
    lut: Dict[str, List[tuple]] = defaultdict(list)
    for rec in iter_index_records():
        lut[str(rec["h"])].append((rec["track_id"], rec["t"]))
    return lut

# --- iTunes helpers ---
def itunes_search(term: str, limit: int = 25) -> List[Dict[str, Any]]:
    url = "https://itunes.apple.com/search"
    params = {"term": term, "media": "music", "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])

def download_preview(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

# --- Routes ---
@app.get("/health")
def health():
    meta = load_meta()
    return {"ok": True, "version": VERSION, "uptime_sec": round(time.time()-START_TIME,1),
            "catalog_tracks": len(meta.get("tracks", {})),
            "index_rows": sum(1 for _ in iter_index_records()) if os.path.isfile(INDEX_JSONL) else 0}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    meta = load_meta()
    recent = list(meta.get("tracks", {}).values())[-10:]
    return templates.TemplateResponse("index.html", {"request": request, "meta": meta, "recent": recent, "version": VERSION})

@app.post("/autocatalog")
def autocatalog(queries: str = Form("james brown, aretha franklin, daft punk, funkadelic, the winstons"),
                per_query: int = Form(20)):
    # Limit cap
    per_query = max(1, min(int(per_query), 50))
    meta = load_meta()
    added = 0
    for q in [x.strip() for x in (queries or "").split(",") if x.strip()]:
        results = itunes_search(q, limit=min(per_query, ITUNES_LIMIT))
        for r in results:
            tid = str(r.get("trackId") or r.get("collectionId") or r.get("trackViewUrl") or r.get("previewUrl"))
            if not tid or "previewUrl" not in r: continue
            if len(meta["tracks"]) >= CATALOG_MAX: break
            if tid in meta["tracks"]: continue
            # download preview
            try:
                raw = download_preview(r["previewUrl"])
                fn = os.path.join(CATALOG_DIR, f"{tid}.m4a")
                with open(fn, "wb") as f: f.write(raw)
                # fingerprint
                pairs = fingerprint_from_audio(raw, fn, SR, CLIP_SECONDS)
                # append to index
                rows = [{"h": int(h), "t": int(t), "track_id": tid} for (h,t) in pairs]
                append_index_rows(rows)
                # save meta
                meta["tracks"][tid] = {
                    "trackName": r.get("trackName"),
                    "artistName": r.get("artistName"),
                    "previewUrl": r.get("previewUrl"),
                    "artworkUrl100": r.get("artworkUrl100"),
                    "trackViewUrl": r.get("trackViewUrl"),
                    "duration_ms": r.get("trackTimeMillis"),
                }
                added += 1
            except Exception:
                continue
    meta["count"] = len(meta["tracks"])
    save_meta(meta)
    return {"status": "ok", "added": added, "total": meta["count"]}

@app.get("/catalog")
def catalog_status():
    meta = load_meta()
    return {"total": meta.get("count", 0), "tracks": meta.get("tracks", {})}

@app.post("/scan", response_class=HTMLResponse)
async def scan(request: Request, file: UploadFile = File(...), start_seconds: int = Form(0)):
    # Only user track is uploaded. We'll scan first QUERY_SECONDS against catalog index.
    if not _rate_limit_ok(request.client.host if request.client else "ip"):
        raise HTTPException(429, "Rate limit")
    raw = await file.read()
    size_mb = len(raw)/(1024*1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"Fail on {size_mb:.1f} MB; limiit {MAX_FILE_SIZE_MB} MB.")
    # fingerprint query
    t0 = time.perf_counter()
    pairs = fingerprint_from_audio(raw, file.filename or "track", SR, QUERY_SECONDS)
    # build lookup
    lut = build_lookup()
    votes_by_track: Dict[str, Counter] = defaultdict(Counter)  # track_id -> Counter(offset)
    matches = 0
    for (h, tq) in pairs:
        lst = lut.get(str(int(h)))
        if not lst: continue
        for (tid, tr) in lst:
            offset = tr - tq
            votes_by_track[tid][offset] += 1
            matches += 1
    # Aggregate best offsets
    ranked = []
    meta = load_meta()
    for tid, counter in votes_by_track.items():
        best_offset, votes = counter.most_common(1)[0]
        info = meta["tracks"].get(str(tid), {})
        ranked.append({
            "track_id": tid,
            "track": info.get("trackName"),
            "artist": info.get("artistName"),
            "preview": info.get("previewUrl"),
            "artwork": info.get("artworkUrl100"),
            "viewUrl": info.get("trackViewUrl"),
            "votes": int(votes),
            "best_offset": int(best_offset)
        })
    ranked.sort(key=lambda x: x["votes"], reverse=True)
    top = [r for r in ranked if r["votes"] >= MATCH_VOTE_THRESHOLD][:50]

    payload = {
        "filename": file.filename,
        "catalog_tracks": len(meta.get("tracks", {})),
        "pairs_query": len(pairs),
        "matches": matches,
        "threshold": MATCH_VOTE_THRESHOLD,
        "results": top,
        "t_sec": round(time.perf_counter()-t0, 3)
    }
    return templates.TemplateResponse("scan.html", {"request": request, "res": payload, "version": VERSION})

@app.post("/scan.json")
async def scan_json(request: Request, file: UploadFile = File(...), start_seconds: int = Form(0)):
    raw = await file.read()
    pairs = fingerprint_from_audio(raw, file.filename or "track", SR, QUERY_SECONDS)
    lut = build_lookup()
    votes_by_track: Dict[str, Counter] = defaultdict(Counter)
    matches = 0
    for (h, tq) in pairs:
        lst = lut.get(str(int(h)))
        if not lst: continue
        for (tid, tr) in lst:
            offset = tr - tq
            votes_by_track[tid][offset] += 1
            matches += 1
    ranked = []
    meta = load_meta()
    for tid, counter in votes_by_track.items():
        best_offset, votes = counter.most_common(1)[0]
        info = meta["tracks"].get(str(tid), {})
        ranked.append({
            "track_id": tid,
            "track": info.get("trackName"),
            "artist": info.get("artistName"),
            "preview": info.get("previewUrl"),
            "artwork": info.get("artworkUrl100"),
            "viewUrl": info.get("trackViewUrl"),
            "votes": int(votes),
            "best_offset": int(best_offset)
        })
    ranked.sort(key=lambda x: x["votes"], reverse=True)
    top = [r for r in ranked if r["votes"] >= MATCH_VOTE_THRESHOLD][:50]
    return {
        "pairs_query": len(pairs),
        "matches": matches,
        "threshold": MATCH_VOTE_THRESHOLD,
        "results": top
    }

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, ok: bool = Depends(require_admin)):
    meta = load_meta()
    return templates.TemplateResponse("admin.html", {"request": request, "version": VERSION, "meta": meta})

