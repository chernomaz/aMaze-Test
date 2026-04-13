"""SQLite database — schema creation and connection helper."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "gui_data.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS mcp_servers (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        name    TEXT NOT NULL UNIQUE,
        url     TEXT NOT NULL,
        transport TEXT NOT NULL DEFAULT 'streamable_http',
        notes   TEXT DEFAULT '',
        env_json TEXT DEFAULT '{}',
        status  TEXT DEFAULT 'unknown',
        tools_json TEXT DEFAULT '[]',
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS agents (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        file_path   TEXT NOT NULL,
        description TEXT DEFAULT '',
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS policies (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        description TEXT DEFAULT '',
        policy_json TEXT NOT NULL,
        created_at  TEXT DEFAULT (datetime('now')),
        updated_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS test_cases (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT NOT NULL UNIQUE,
        description  TEXT DEFAULT '',
        policy_name  TEXT NOT NULL,
        agent_name   TEXT NOT NULL,
        prompt       TEXT NOT NULL,
        expected_pass INTEGER NOT NULL DEFAULT 1,
        created_at   TEXT DEFAULT (datetime('now')),
        updated_at   TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS suites (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT NOT NULL UNIQUE,
        description TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS suite_cases (
        suite_id     INTEGER NOT NULL REFERENCES suites(id) ON DELETE CASCADE,
        test_case_id INTEGER NOT NULL REFERENCES test_cases(id) ON DELETE CASCADE,
        position     INTEGER NOT NULL DEFAULT 0,
        enabled      INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (suite_id, test_case_id)
    );

    CREATE TABLE IF NOT EXISTS suite_runs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        suite_id   INTEGER NOT NULL REFERENCES suites(id),
        suite_name TEXT NOT NULL,
        started_at TEXT DEFAULT (datetime('now')),
        finished_at TEXT,
        status     TEXT DEFAULT 'running',
        pass_count INTEGER DEFAULT 0,
        fail_count INTEGER DEFAULT 0,
        xfail_count INTEGER DEFAULT 0,
        xpass_count INTEGER DEFAULT 0,
        total_count INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS test_runs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        suite_run_id  INTEGER REFERENCES suite_runs(id),
        test_case_name TEXT NOT NULL,
        policy_name   TEXT NOT NULL,
        agent_name    TEXT NOT NULL,
        prompt        TEXT NOT NULL,
        expected_pass INTEGER NOT NULL DEFAULT 1,
        started_at    TEXT DEFAULT (datetime('now')),
        finished_at   TEXT,
        outcome       TEXT,
        audit_json_path TEXT,
        audit_html_path TEXT,
        log_output    TEXT DEFAULT ''
    );
    """)
    conn.commit()
    conn.close()
