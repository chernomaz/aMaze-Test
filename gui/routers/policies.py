import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from gui.database import get_conn
from gui.models import PolicyIn

router = APIRouter(prefix="/api/policies", tags=["policies"])

POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "policies"


def _sync_from_disk():
    """Import any .json files from policies/ dir that aren't in the DB yet."""
    if not POLICIES_DIR.exists():
        return
    conn = get_conn()
    for f in sorted(POLICIES_DIR.glob("*.json")):
        name = f.stem
        existing = conn.execute("SELECT id FROM policies WHERE name=?", (name,)).fetchone()
        if not existing:
            try:
                raw = f.read_text(encoding="utf-8")
                json.loads(raw)  # validate
                conn.execute(
                    "INSERT OR IGNORE INTO policies (name,policy_json) VALUES (?,?)",
                    (name, raw)
                )
            except Exception:
                pass
    conn.commit()
    conn.close()


@router.get("")
def list_policies():
    _sync_from_disk()
    conn = get_conn()
    rows = conn.execute("SELECT * FROM policies ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/{name}")
def get_policy(name: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM policies WHERE name=?", (name,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Policy not found")
    return dict(row)


@router.post("", status_code=201)
def create_policy(body: PolicyIn):
    try:
        json.loads(body.policy_json)
    except Exception:
        raise HTTPException(400, "policy_json is not valid JSON")
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO policies (name,description,policy_json) VALUES (?,?,?)",
            (body.name, body.description, body.policy_json)
        )
        conn.commit()
        _write_policy_file(body.name, body.policy_json)
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()
    return {"ok": True}


@router.put("/{name}")
def update_policy(name: str, body: PolicyIn):
    try:
        json.loads(body.policy_json)
    except Exception:
        raise HTTPException(400, "policy_json is not valid JSON")
    conn = get_conn()
    conn.execute(
        "UPDATE policies SET description=?,policy_json=?,updated_at=datetime('now') WHERE name=?",
        (body.description, body.policy_json, name)
    )
    conn.commit()
    conn.close()
    _write_policy_file(name, body.policy_json)
    return {"ok": True}


@router.delete("/{name}")
def delete_policy(name: str):
    conn = get_conn()
    conn.execute("DELETE FROM policies WHERE name=?", (name,))
    conn.commit()
    conn.close()
    p = POLICIES_DIR / f"{name}.json"
    if p.exists():
        p.unlink()
    return {"ok": True}


def _write_policy_file(name: str, policy_json: str):
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    (POLICIES_DIR / f"{name}.json").write_text(policy_json, encoding="utf-8")
