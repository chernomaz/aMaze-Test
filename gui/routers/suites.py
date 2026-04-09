from fastapi import APIRouter, HTTPException
from gui.database import get_conn
from gui.models import SuiteIn

router = APIRouter(prefix="/api/suites", tags=["suites"])


@router.get("")
def list_suites():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM suites ORDER BY name").fetchall()
    result = []
    for row in rows:
        suite = dict(row)
        cases = conn.execute(
            "SELECT tc.name, sc.position, sc.enabled "
            "FROM suite_cases sc JOIN test_cases tc ON sc.test_case_id=tc.id "
            "WHERE sc.suite_id=? ORDER BY sc.position",
            (suite["id"],)
        ).fetchall()
        suite["test_cases"] = [dict(c) for c in cases]
        result.append(suite)
    conn.close()
    return result


@router.get("/{name}")
def get_suite(name: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM suites WHERE name=?", (name,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Suite not found")
    suite = dict(row)
    cases = conn.execute(
        "SELECT tc.name, sc.position, sc.enabled "
        "FROM suite_cases sc JOIN test_cases tc ON sc.test_case_id=tc.id "
        "WHERE sc.suite_id=? ORDER BY sc.position",
        (suite["id"],)
    ).fetchall()
    suite["test_cases"] = [dict(c) for c in cases]
    conn.close()
    return suite


@router.post("", status_code=201)
def create_suite(body: SuiteIn):
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO suites (name,description) VALUES (?,?)",
            (body.name, body.description)
        )
        suite_id = cur.lastrowid
        for i, tc_name in enumerate(body.test_case_names):
            tc_row = conn.execute("SELECT id FROM test_cases WHERE name=?", (tc_name,)).fetchone()
            if tc_row:
                conn.execute(
                    "INSERT OR IGNORE INTO suite_cases (suite_id,test_case_id,position) VALUES (?,?,?)",
                    (suite_id, tc_row["id"], i)
                )
        conn.commit()
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()
    return {"ok": True}


@router.put("/{name}")
def update_suite(name: str, body: SuiteIn):
    conn = get_conn()
    row = conn.execute("SELECT id FROM suites WHERE name=?", (name,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Suite not found")
    suite_id = row["id"]
    conn.execute("UPDATE suites SET description=? WHERE id=?", (body.description, suite_id))
    conn.execute("DELETE FROM suite_cases WHERE suite_id=?", (suite_id,))
    for i, tc_name in enumerate(body.test_case_names):
        tc_row = conn.execute("SELECT id FROM test_cases WHERE name=?", (tc_name,)).fetchone()
        if tc_row:
            conn.execute(
                "INSERT OR IGNORE INTO suite_cases (suite_id,test_case_id,position) VALUES (?,?,?)",
                (suite_id, tc_row["id"], i)
            )
    conn.commit()
    conn.close()
    return {"ok": True}


@router.delete("/{name}")
def delete_suite(name: str):
    conn = get_conn()
    conn.execute("DELETE FROM suites WHERE name=?", (name,))
    conn.commit()
    conn.close()
    return {"ok": True}
