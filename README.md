# SampleDetector v1.4.0 · AutoCatalog

No reference uploads. Builds an index by downloading 30s preview clips from iTunes Search API for your queries.

Endpoints:
- POST /autocatalog  (form: queries="artist1, artist2", per_query=20)
- POST /scan         (form-data: file=your_track)
- POST /scan.json    (API)
- GET  /catalog, /health

Env knobs: CATALOG_MAX, ITUNES_LIMIT, SR, CLIP_SECONDS, QUERY_SECONDS, PEAKS_PER_FRAME, FANOUT, MATCH_VOTE_THRESHOLD

Render deploy: Python 3.11.9; lightweight deps.
