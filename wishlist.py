"""SQLite-backed wishlist store."""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "wishlist.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wishlist (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name      TEXT NOT NULL,
                card_name        TEXT NOT NULL,
                preferred_rarity TEXT,
                preferred_set    TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def get_all():
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, player_name, card_name, preferred_rarity, preferred_set "
            "FROM wishlist ORDER BY player_name, card_name"
        ).fetchall()
        return [dict(r) for r in rows]


def add_entry(player_name, card_name, preferred_rarity=None, preferred_set=None):
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO wishlist (player_name, card_name, preferred_rarity, preferred_set) "
            "VALUES (?, ?, ?, ?)",
            (
                player_name.strip(),
                card_name.strip(),
                preferred_rarity.strip() if preferred_rarity else None,
                preferred_set.strip() if preferred_set else None,
            ),
        )
        return cur.lastrowid


def remove_entry(entry_id: int):
    with _conn() as conn:
        conn.execute("DELETE FROM wishlist WHERE id = ?", (entry_id,))
