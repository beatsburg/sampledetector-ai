import os, io, json, tempfile, time, zipfile, math
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import imageio_ffmpeg
from annoy import AnnoyIndex

# ffmpeg path for pydub
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

VERSION = "2.0-alpha"
SR = 22050
WINDOW_SEC = 2.0          # sliding window size
HOP_SEC = 0.5             # step between windows
EMB_DIM = 64              # embedding dimensionality after pooling
MAX_REF_FILES = 200       # keep alpha bounded
MAX_REF_SECONDS = 120     # per ref file analyze up to N seconds
MAX_TRACK_SECONDS = 120   # per query track
PITCH_STEPS = list(range(-4, 5))  # -4..+4 semitones
TEMPO_FACTORS = [0.9, 1.0, 1.1]   # -10%, 0, +10%

INDEX_DIR = "ref_index"
META_PATH = os.path.join(INDEX_DIR, "meta.json")
ANNOY_PATH = os.path.join(INDEX_DIR, "annoy_index.ann")
VECTOR_PATH = os.path.join(INDEX_DIR, "vectors.npy")

os.makedirs(INDEX_DIR, exist_ok=True)
templates = Jinja2Templates(directory="app/templates")

app = FastAPI(title="SampleDetector v2.0-alpha")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

def load_audio_bytes(raw: bytes, filename: str, sr: int = SR, max_seconds: float = None):
    # decode with librosa (fallback to pydub if needed)
    ext = os.path.splitext(filename)[1].lower()
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        y, _sr = librosa.load(tmp_path, sr=sr, mono=True)
        if max_seconds:
            y = y[: int(max_seconds * sr)]
        return y, sr
    except Exception:
        audio = AudioSegment.from_file(io.BytesIO(raw), format=ext.replace(".", "") if ext else None)
        if max_seconds and len(audio) > max_seconds * 1000:
            audio = audio[: int(max_seconds * 1000)]
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        y, _sr = sf.read(buf)
        if _sr != sr:
            y = librosa.resample(y, orig_sr=_sr, target_sr=sr, res_type="scipy")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y, sr
    finally:
        try:
            if 'tmp_path' in locals():
                os.remove(tmp_path)
        except Exception:
            pass

def embed_signal(y: np.ndarray, sr: int) -> np.ndarray:
    # Compute chroma_cqt + mfcc, then pool to fixed size
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=int(sr*0.01))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=int(sr*0.01))
    # normalize
    chroma = librosa.util.normalize(chroma, axis=1)
    mfcc = librosa.util.normalize(mfcc, axis=1)
    # temporal pooling: mean + std
    feat = np.concatenate([chroma.mean(axis=1), chroma.std(axis=1), mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0)
    # If needed, pad/truncate to EMB_DIM
    if feat.shape[0] < EMB_DIM:
        feat = np.pad(feat, (0, EMB_DIM - feat.shape[0]))
    else:
        feat = feat[:EMB_DIM]
    # L2 normalize for cosine similarity
    norm = np.linalg.norm(feat) + 1e-9
    return (feat / norm).astype(np.float32)

def pitch_shift(y: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    if semitones == 0:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones, res_type="scipy")

def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    if abs(rate - 1.0) < 1e-6:
        return y
    return librosa.effects.time_stretch(y, rate=rate)

def sliding_windows(y: np.ndarray, sr: int, win_s: float, hop_s: float):
    win = int(sr * win_s)
    hop = int(sr * hop_s)
    for start in range(0, max(1, len(y) - win + 1), hop):
        yield start, y[start:start+win]

def build_index_from_zip(raw: bytes) -> Dict[str, Any]:
    # Unzip to temp, iterate files, build vectors
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extractall(tmpdir)

    vectors = []
    meta = []
    total_files = 0

    for root, _, files in os.walk(tmpdir):
        for nm in files:
            if total_files >= MAX_REF_FILES: break
            if not nm.lower().endswith((".mp3",".wav",".m4a",".ogg",".aiff",".aif",".flac")):
                continue
            total_files += 1
            full = os.path.join(root, nm)
            with open(full, "rb") as f:
                rawf = f.read()
            y, sr = load_audio_bytes(rawf, nm, sr=SR, max_seconds=MAX_REF_SECONDS)
            # windows
            for start, chunk in sliding_windows(y, sr, WINDOW_SEC, HOP_SEC):
                vec = embed_signal(chunk, sr)
                vectors.append(vec)
                meta.append({
                    "file": nm,
                    "start": start / sr,
                    "duration": WINDOW_SEC
                })

    if not vectors:
        raise HTTPException(400, "ZIP ei sisaldanud ühtegi sobivat audiofaili.")

    vecs = np.stack(vectors, axis=0)
    # Save vectors + meta
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(VECTOR_PATH, vecs)
    with open(META_PATH, "w") as f:
        json.dump({"meta": meta, "sr": SR, "window_s": WINDOW_SEC, "hop_s": HOP_SEC}, f)

    # Build Annoy
    t = AnnoyIndex(EMB_DIM, 'angular')
    for i in range(vecs.shape[0]):
        t.add_item(i, vecs[i])
    t.build(20)  # n_trees
    t.save(ANNOY_PATH)

    return {"vectors": int(vecs.shape[0]), "ref_files": total_files}

def ensure_index_loaded() -> Tuple[AnnoyIndex, np.ndarray, Dict[str,Any]]:
    if not (os.path.isfile(ANNOY_PATH) and os.path.isfile(VECTOR_PATH) and os.path.isfile(META_PATH)):
        raise HTTPException(404, "Indeks puudub. Laadi referentsid /index-samples kaudu.")
    vecs = np.load(VECTOR_PATH)
    with open(META_PATH,"r") as f:
        meta = json.load(f)
    t = AnnoyIndex(EMB_DIM, 'angular')
    t.load(ANNOY_PATH)
    return t, vecs, meta

def analyze_track_against_index(raw: bytes, filename: str, start_s: float, max_seconds: float, topk: int = 5) -> Dict[str, Any]:
    t_ann, vecs, meta = ensure_index_loaded()

    y, sr = load_audio_bytes(raw, filename, sr=SR, max_seconds=max_seconds)
    results = []
    # iterate variations
    for rate in TEMPO_FACTORS:
        y_t = time_stretch(y, rate)
        for st in PITCH_STEPS:
            y_v = pitch_shift(y_t, sr, st) if st != 0 else y_t
            for start, chunk in sliding_windows(y_v, sr, WINDOW_SEC, HOP_SEC):
                if start / sr + WINDOW_SEC < start_s:
                    continue
                vec = embed_signal(chunk, sr)
                idxs = t_ann.get_nns_by_vector(vec, topk, include_distances=True)
                for i, dist in zip(idxs[0], idxs[1]):
                    ref = meta["meta"][i]
                    sim = 1.0 - (dist**2)/2.0  # angular to cosine approx
                    if sim >= 0.80:  # threshold for alpha
                        results.append({
                            "query_time": round(start/sr, 2),
                            "ref_file": ref["file"],
                            "ref_time": round(ref["start"], 2),
                            "pitch_steps": st,
                            "tempo_factor": rate,
                            "similarity": round(float(sim), 3)
                        })
    # Group by (ref_file), take best runs (simple)
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return {
        "filename": filename,
        "query_seconds": round(len(y)/sr, 2),
        "matches": results[:200]
    }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "version": VERSION})

@app.get("/health")
def health():
    ok = os.path.isfile(ANNOY_PATH)
    return {"ok": True, "version": VERSION, "index_ready": ok}

@app.post("/index-samples")
def index_samples(zipfile_upload: UploadFile = File(...)):
    raw = zipfile_upload.file.read()
    if not raw:
        raise HTTPException(400, "Tühi upload.")
    try:
        info = build_index_from_zip(raw)
        return {"status": "ok", "info": info}
    except zipfile.BadZipFile:
        raise HTTPException(400, "Fail ei ole korrektne ZIP.")
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/sample-origin", response_class=HTMLResponse)
def sample_origin_html(request: Request, track: UploadFile = File(...), start_seconds: float = Form(0)):
    raw = track.file.read()
    res = analyze_track_against_index(raw, track.filename or "track", start_seconds, MAX_TRACK_SECONDS, topk=5)
    return templates.TemplateResponse("sample_origin.html", {"request": request, "res": res, "version": VERSION})

@app.post("/sample-origin.json")
def sample_origin_json(track: UploadFile = File(...), start_seconds: float = Form(0)):
    raw = track.file.read()
    res = analyze_track_against_index(raw, track.filename or "track", start_seconds, MAX_TRACK_SECONDS, topk=5)
    return res

@app.get("/index/meta")
def index_meta():
    if not os.path.isfile(META_PATH):
        return PlainTextResponse("Indeks puudub.", status_code=404)
    with open(META_PATH, "r") as f:
        return json.load(f)

@app.get("/index/clear")
def index_clear():
    n=0
    for p in [ANNOY_PATH, VECTOR_PATH, META_PATH]:
        if os.path.isfile(p):
            os.remove(p); n+=1
    return {"cleared_files": n}
