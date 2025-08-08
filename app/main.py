from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_audio(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        f.write(contents)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "filename": file.filename,
        "message": "Fail laeti üles ja salvestati."
    })

@app.get("/uploads/{filename}")
async def get_file(filename: str):
    return FileResponse(path=os.path.join(UPLOAD_FOLDER, filename))
