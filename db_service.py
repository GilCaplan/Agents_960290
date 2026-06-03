"""
Local SQLite chat store — drop-in replacement for the Supabase backend.

The project previously persisted chats/messages in Supabase (PostgreSQL), which
requires a project URL + API key. Since we are running fully locally, this module
uses the bundled `agri_advisor.db` SQLite file with the identical schema, exposing
the same helper functions main.py already calls.
"""

import sqlite3
import threading

DB_PATH = "agri_advisor.db"
_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _lock, _conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS chats ("
                  "chat_id TEXT PRIMARY KEY, user_name TEXT, title TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS messages ("
                  "id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT)")


def get_chat(chat_id: str):
    with _lock, _conn() as c:
        row = c.execute("SELECT chat_id FROM chats WHERE chat_id=?", (chat_id,)).fetchone()
        return dict(row) if row else None


def create_chat(chat_id: str, user_name: str, title: str):
    with _lock, _conn() as c:
        c.execute("INSERT OR IGNORE INTO chats (chat_id, user_name, title) VALUES (?,?,?)",
                  (chat_id, user_name, title))


def get_history(chat_id: str):
    with _lock, _conn() as c:
        rows = c.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY id",
                         (chat_id,)).fetchall()
        return [dict(r) for r in rows]


def save_messages(chat_id: str, user_msg: str, bot_msg: str):
    with _lock, _conn() as c:
        c.executemany("INSERT INTO messages (chat_id, role, content) VALUES (?,?,?)",
                      [(chat_id, "user", user_msg), (chat_id, "bot", bot_msg)])


def get_user_chats(user_name: str):
    with _lock, _conn() as c:
        rows = c.execute("SELECT chat_id, title FROM chats WHERE user_name=?",
                         (user_name,)).fetchall()
        return [dict(r) for r in rows]


def delete_chat(chat_id: str):
    with _lock, _conn() as c:
        c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        c.execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))
