import sqlite3
from typing import List, Optional, Dict, Tuple

DB_PATH = "chat_threads_auth.sqlite"

def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            index_dir TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            role TEXT,
            content TEXT,
            ts TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(thread_id) REFERENCES threads(id)
        )
        """
    )
    # index for quick ownership lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id)")
    con.commit()
    con.close()

def create_user_row(user_id: str, email: str, salt: str, password_hash: str) -> bool:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO users (id, email, salt, password_hash) VALUES (?, ?, ?, ?)",
            (user_id, email, salt, password_hash),
        )
        con.commit()
        ok = True
    except sqlite3.IntegrityError:
        ok = False
    finally:
        con.close()
    return ok

def read_user_by_email(email: str) -> Optional[Tuple[str, str, str, str]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, email, salt, password_hash FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    con.close()
    return row

def upsert_thread(user_id: str, thread_id: str, title: str, index_dir: str) -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO threads (id, user_id, title, index_dir, created_at, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            user_id=excluded.user_id,
            title=excluded.title,
            index_dir=excluded.index_dir,
            updated_at=datetime('now')
        """,
        (thread_id, user_id, title, index_dir),
    )
    con.commit()
    con.close()

def insert_message(thread_id: str, role: str, content: str) -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)", (thread_id, role, content))
    cur.execute("UPDATE threads SET updated_at=datetime('now') WHERE id=?", (thread_id,))
    con.commit()
    con.close()

def load_threads(user_id: str, limit: int = 30) -> List[Dict[str, str]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, title, index_dir FROM threads
        WHERE user_id=?
        ORDER BY updated_at DESC LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cur.fetchall()
    con.close()
    return [{"id": r[0], "title": r[1], "index_dir": r[2]} for r in rows]

def is_thread_owned_by(user_id: str, thread_id: str) -> bool:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT 1 FROM threads WHERE id=? AND user_id=? LIMIT 1", (thread_id, user_id))
    row = cur.fetchone()
    con.close()
    return row is not None

def load_messages_for_user(user_id: str, thread_id: str) -> List[Dict[str, str]]:
    """Return messages only if the thread belongs to the user; else empty list."""
    if not is_thread_owned_by(user_id, thread_id):
        return []
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        SELECT m.role, m.content
        FROM messages m
        JOIN threads t ON t.id = m.thread_id
        WHERE m.thread_id=? AND t.user_id=?
        ORDER BY m.id ASC
        """,
        (thread_id, user_id),
    )
    rows = cur.fetchall()
    con.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

# Legacy function (no user check) — prefer load_messages_for_user
def load_messages(thread_id: str) -> List[Dict[str, str]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT role, content FROM messages WHERE thread_id=? ORDER BY id ASC", (thread_id,))
    rows = cur.fetchall()
    con.close()
    return [{"role": r[0], "content": r[1]} for r in rows]
