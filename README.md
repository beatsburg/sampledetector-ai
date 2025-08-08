
# SampleGuard

## Deploy to Render

1. Push this repo to GitHub
2. Go to https://dashboard.render.com/
3. Create a new Web Service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
   - Region: Frankfurt (or closest)
   - Port: 10000 (set as env var)
4. Done! Open the public URL to access the uploader
5. Outputs are stored in `/uploads` (PDF + analysis JSON)
