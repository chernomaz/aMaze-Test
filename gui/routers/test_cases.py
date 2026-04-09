from fastapi import APIRouter, HTTPException
from gui.database import get_conn
from gui.models import TestCaseIn

router = APIRouter(prefix="/api/test-cases", tags=["test-cases"])


@router.get("")
def list_test_cases():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM test_cases ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/{name}")
def get_test_case(name: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM test_cases WHERE name=?", (name,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Test case not found")
    return dict(row)


@router.post("", status_code=201)
def create_test_case(body: TestCaseIn):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO test_cases (name,description,policy_name,agent_name,prompt,expected_pass) "
            "VALUES (?,?,?,?,?,?)",
            (body.name, body.description, body.policy_name, body.agent_name,
             body.prompt, 1 if body.expected_pass else 0)
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()
    return {"ok": True}


@router.put("/{name}")
def update_test_case(name: str, body: TestCaseIn):
    conn = get_conn()
    conn.execute(
        "UPDATE test_cases SET description=?,policy_name=?,agent_name=?,prompt=?,"
        "expected_pass=?,updated_at=datetime('now') WHERE name=?",
        (body.description, body.policy_name, body.agent_name, body.prompt,
         1 if body.expected_pass else 0, name)
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@router.delete("/{name}")
def delete_test_case(name: str):
    conn = get_conn()
    conn.execute("DELETE FROM test_cases WHERE name=?", (name,))
    conn.commit()
    conn.close()
    return {"ok": True}
