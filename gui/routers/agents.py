from fastapi import APIRouter, HTTPException
from gui.database import get_conn
from gui.models import AgentIn

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("")
def list_agents():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM agents ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("", status_code=201)
def create_agent(body: AgentIn):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO agents (name,file_path,description) VALUES (?,?,?)",
            (body.name, body.file_path, body.description)
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()
    return {"ok": True}


@router.put("/{name}")
def update_agent(name: str, body: AgentIn):
    conn = get_conn()
    conn.execute(
        "UPDATE agents SET file_path=?,description=? WHERE name=?",
        (body.file_path, body.description, name)
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@router.delete("/{name}")
def delete_agent(name: str):
    conn = get_conn()
    conn.execute("DELETE FROM agents WHERE name=?", (name,))
    conn.commit()
    conn.close()
    return {"ok": True}
