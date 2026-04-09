"""aMazeTest GUI server — FastAPI entry point.

Run with:
    cd /data/cloude/aMazeTest
    /data/venv/bin/uvicorn gui.server:app --reload --port 8080
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from gui.database import init_db
from gui.routers import agents, mcp_servers, policies, runs, suites, test_cases

app = FastAPI(title="aMazeTest GUI", version="0.1.0")

# Init DB on startup
@app.on_event("startup")
def startup():
    init_db()


# Routers
app.include_router(agents.router)
app.include_router(mcp_servers.router)
app.include_router(policies.router)
app.include_router(runs.router)
app.include_router(suites.router)
app.include_router(test_cases.router)

# Serve audit HTML reports from audit_logs/
AUDIT_DIR = Path(__file__).resolve().parent.parent / "audit_logs"

@app.get("/audit/{filename}")
def serve_audit(filename: str):
    path = AUDIT_DIR / filename
    if not path.exists() or not path.suffix == ".html":
        from fastapi import HTTPException
        raise HTTPException(404)
    return FileResponse(path, media_type="text/html")


# Serve the SPA
SPA_PATH = Path(__file__).resolve().parent / "static" / "index.html"

@app.get("/", response_class=HTMLResponse)
@app.get("/{path:path}", response_class=HTMLResponse)
def spa(path: str = ""):
    if SPA_PATH.exists():
        return HTMLResponse(SPA_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>aMazeTest GUI</h1><p>Frontend not built yet.</p>")
