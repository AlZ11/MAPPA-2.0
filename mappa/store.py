"""Evidence store: one SQLite file plus a folder of frozen raw files.

Raw files (web pages, extracted policy text) are saved gzipped under data/raw/ab/<sha256>.gz,
named by the SHA-256 fingerprint of their uncompressed bytes. The same content is stored once,
and anyone can later prove what a page said on the day it was fetched.
"""

import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
from dataclasses import astuple, fields
from datetime import datetime, timezone
from pathlib import Path

from .vocab import Claim

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run TEXT PRIMARY KEY, kind TEXT, started_at TEXT, git_commit TEXT, params TEXT
);
CREATE TABLE IF NOT EXISTS fetches (
    run TEXT, app_id TEXT, kind TEXT, url TEXT, final_url TEXT, status INTEGER,
    content_type TEXT, sha256 TEXT, bytes INTEGER, fetched_at TEXT, attempts INTEGER, note TEXT,
    UNIQUE (run, app_id, kind)
);
CREATE TABLE IF NOT EXISTS apps (
    run TEXT, app_id TEXT, found_via TEXT, rank INTEGER, title TEXT, developer TEXT, genre TEXT,
    installs INTEGER, updated TEXT, version TEXT, policy_url TEXT, label_status TEXT,
    label_sha TEXT, text_sha TEXT, policy_date TEXT, policy_flags TEXT,
    PRIMARY KEY (run, app_id)
);
CREATE TABLE IF NOT EXISTS reads (
    id INTEGER PRIMARY KEY, run TEXT, app_id TEXT, source TEXT, extractor TEXT, model TEXT,
    prompt_version TEXT, rep INTEGER, input_sha TEXT, seconds REAL, prompt_tokens INTEGER,
    completion_tokens INTEGER, n_claims INTEGER, dropped TEXT, status TEXT, created_at TEXT,
    UNIQUE (run, app_id, source, extractor, model, prompt_version, rep)
);
CREATE TABLE IF NOT EXISTS claims (
    read_id INTEGER, kind TEXT, polarity TEXT, category TEXT, subtype TEXT, action TEXT,
    practice TEXT, recipient TEXT, detail TEXT, purposes TEXT, optional INTEGER, quote TEXT,
    seg INTEGER, start INTEGER, "end" INTEGER
);
CREATE INDEX IF NOT EXISTS claims_read ON claims (read_id);
"""

# A fetch with one of these statuses is finished: 200 = got it, 404 = the thing does not exist.
FINAL_STATUSES = (200, 404)
CLAIM_COLUMNS = [f.name for f in fields(Claim)]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


class Store:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "mappa.db")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------------ raw files
    def _blob_path(self, sha: str) -> Path:
        return self.root / "raw" / sha[:2] / f"{sha}.gz"

    def put_blob(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        path = self._blob_path(sha)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(data, compresslevel=6, mtime=0))
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written file under the real name
        return sha

    def get_blob(self, sha: str) -> bytes:
        return gzip.decompress(self._blob_path(sha).read_bytes())

    # ------------------------------------------------------------------ runs and fetches
    def start_run(self, run: str, kind: str, params: dict) -> None:
        self.db.execute("INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, ?)",
                        (run, kind, now(), git_commit(), json.dumps(params, sort_keys=True)))
        self.db.commit()

    def run_kind(self, run: str) -> str | None:
        row = self.db.execute("SELECT kind FROM runs WHERE run = ?", (run,)).fetchone()
        return row["kind"] if row else None

    def fetch(self, run: str, app_id: str, kind: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM fetches WHERE run = ? AND app_id = ? AND kind = ?",
                               (run, app_id, kind)).fetchone()

    def pending(self, run: str, app_id: str, kind: str, max_attempts: int = 3) -> bool:
        """True if this item still needs fetching. This is what makes a crawl resumable."""
        row = self.fetch(run, app_id, kind)
        if row is None:
            return True
        return row["status"] not in FINAL_STATUSES and row["attempts"] < max_attempts

    def record_fetch(self, run: str, app_id: str, kind: str, *, url: str, final_url: str, status: int,
                     content_type: str, body: bytes, note: str | None = None) -> str | None:
        sha = self.put_blob(body) if body else None
        self.db.execute(
            """INSERT INTO fetches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
               ON CONFLICT (run, app_id, kind) DO UPDATE SET
                 url = excluded.url, final_url = excluded.final_url, status = excluded.status,
                 content_type = excluded.content_type, sha256 = excluded.sha256, bytes = excluded.bytes,
                 fetched_at = excluded.fetched_at, note = excluded.note, attempts = attempts + 1""",
            (run, app_id, kind, url, final_url, status, content_type, sha, len(body), now(), note),
        )
        self.db.commit()
        return sha

    def body(self, run: str, app_id: str, kind: str) -> bytes | None:
        """The frozen bytes of a successful fetch, or None."""
        row = self.fetch(run, app_id, kind)
        if row is None or row["status"] != 200 or not row["sha256"]:
            return None
        return self.get_blob(row["sha256"])

    # ------------------------------------------------------------------ apps
    def add_app(self, run: str, app_id: str, found_via: str, rank: int) -> None:
        """Register an app in a run. The first way it was found is kept."""
        self.db.execute("INSERT OR IGNORE INTO apps (run, app_id, found_via, rank) VALUES (?, ?, ?, ?)",
                        (run, app_id, found_via, rank))
        self.db.commit()

    def update_app(self, run: str, app_id: str, **cols) -> None:
        if not cols:
            return
        sets = ", ".join(f"{k} = ?" for k in cols)
        self.db.execute(f"UPDATE apps SET {sets} WHERE run = ? AND app_id = ?", (*cols.values(), run, app_id))
        self.db.commit()

    def apps(self, run: str, genres: set[str] | None = None) -> list[sqlite3.Row]:
        rows = self.db.execute("SELECT * FROM apps WHERE run = ? ORDER BY rowid", (run,)).fetchall()
        return [r for r in rows if genres is None or r["genre"] in genres]

    # ------------------------------------------------------------------ reads and claims
    def save_read(self, run: str, app_id: str, source: str, extractor: str, claims: list[Claim], *,
                  input_sha: str, stats: dict, model: str = "", prompt_version: str = "", rep: int = 0,
                  status: str = "ok") -> int:
        """Store one reading of one document. Re-reading with the same key replaces the old rows."""
        key = (run, app_id, source, extractor, model, prompt_version, rep)
        where = "run = ? AND app_id = ? AND source = ? AND extractor = ? AND model = ? AND prompt_version = ? AND rep = ?"
        old = self.db.execute(f"SELECT id FROM reads WHERE {where}", key).fetchone()
        if old:
            self.db.execute("DELETE FROM claims WHERE read_id = ?", (old["id"],))
            self.db.execute("DELETE FROM reads WHERE id = ?", (old["id"],))
        cur = self.db.execute(
            "INSERT INTO reads (run, app_id, source, extractor, model, prompt_version, rep, input_sha, seconds,"
            " prompt_tokens, completion_tokens, n_claims, dropped, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*key, input_sha, stats.get("seconds"), stats.get("prompt_tokens"), stats.get("completion_tokens"),
             len(claims), json.dumps(stats.get("dropped", {}), sort_keys=True), status, now()),
        )
        read_id = cur.lastrowid
        rows = []
        for c in claims:
            values = list(astuple(c))
            values[CLAIM_COLUMNS.index("purposes")] = json.dumps(list(c.purposes))
            rows.append((read_id, *values))
        cols = ", ".join(f'"{c}"' for c in CLAIM_COLUMNS)
        self.db.executemany(f"INSERT INTO claims (read_id, {cols}) VALUES ({', '.join('?' * (len(CLAIM_COLUMNS) + 1))})", rows)
        self.db.commit()
        return read_id

    def claims(self, run: str, app_id: str, source: str | None = None) -> list[sqlite3.Row]:
        sql = ("SELECT reads.source, reads.extractor, reads.model, reads.rep, claims.* FROM claims"
               " JOIN reads ON reads.id = claims.read_id WHERE reads.run = ? AND reads.app_id = ?")
        args = [run, app_id]
        if source:
            sql += " AND reads.source = ?"
            args.append(source)
        return self.db.execute(sql + " ORDER BY reads.id, claims.rowid", args).fetchall()
