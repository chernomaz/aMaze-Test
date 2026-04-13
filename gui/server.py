"""aMazeTest GUI server — FastAPI entry point.

Must be run from the project root directory:
    cd /path/to/aMazeTest
    /path/to/venv/bin/uvicorn gui.server:app --reload --port 8080 --host 0.0.0.0
"""
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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

# File browser — returns directory listing for the file-picker modal
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@app.get("/api/browse")
def browse(path: str = Query(default="")):
    target = (PROJECT_ROOT / path).resolve() if path else PROJECT_ROOT.resolve()
    # Prevent escaping the filesystem root (no restriction, server-local only)
    if not target.exists() or not target.is_dir():
        return JSONResponse({"error": "Not a directory"}, status_code=400)
    entries = []
    try:
        parent = str(target.parent.relative_to(PROJECT_ROOT)) if target != PROJECT_ROOT else None
    except ValueError:
        parent = None
    for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        try:
            rel = str(item.relative_to(PROJECT_ROOT))
        except ValueError:
            continue
        entries.append({"name": item.name, "path": rel, "is_dir": item.is_dir()})
    return {"current": str(target.relative_to(PROJECT_ROOT)) if target != PROJECT_ROOT else "", "parent": parent, "entries": entries}


# Serve audit HTML reports from reports/
AUDIT_DIR = Path(__file__).resolve().parent.parent / "reports"

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
