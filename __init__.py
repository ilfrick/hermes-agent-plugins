"""Ollama Memory Provider — MemoryProvider interface.

Local-first semantic memory using Ollama embeddings (default: qwen3-embedding:4b)
and SQLite for storage. No external dependencies beyond Ollama HTTP API and stdlib.

Config (config.yaml → memory.ollama_memory):
  endpoint: "http://192.168.10.20:11434"   # Ollama API base URL
  model: "qwen3-embedding:4b"              # Embedding model name
  collection: "hermes_memory"              # Logical collection / table suffix
  top_k: 8                                 # Default search results
  similarity_threshold: 0.25               # Cosine similarity cutoff (0-1)
  max_context_tokens: 1200                 # Max tokens in prefetch context block
  auto_embed_writes: true                  # Mirror built-in memory.add to Ollama store
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.memory_provider import MemoryProvider, spawn_context_thread
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://192.168.10.20:11434"
_DEFAULT_MODEL = "qwen3-embedding:4b"
_DEFAULT_COLLECTION = "hermes_memory"
_DEFAULT_TOP_K = 8
_DEFAULT_SIMILARITY_THRESHOLD = 0.25
_DEFAULT_MAX_CONTEXT_TOKENS = 1200
_DEFAULT_KEEP_ALIVE = "10m"
_DEFAULT_PREFETCH_EMBEDDING_TIMEOUT = 5.0
_EMBED_DIM = 2560  # qwen3-embedding:4b dimension
_IMPORT_CHUNK_CHARS = 480
_PROFILE_IMPORT_VERSION = 2
_EMBED_BATCH_SIZE = 4
_PROFILE_MEMORY_SOURCES = (
    (("memories/MEMORY.md", "MEMORY.md"), "factual", 0.8),
    (("memories/USER.md", "USER.md"), "preference", 0.8),
    (("SOUL.md",), "instruction", 0.9),
)


def _split_for_embedding(text: str, max_chars: int = _IMPORT_CHUNK_CHARS) -> List[Tuple[str, str]]:
    """Split Markdown into bounded paragraph/fact chunks and retain section labels."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized or max_chars < 8:
        return []
    chunks: List[Tuple[str, str]] = []
    section = "General"
    current: List[str] = []

    def flush() -> None:
        if current:
            body = "\n".join(current)
            label = section[:max(1, max_chars // 3)]
            content = f"{label}\n{body}"
            chunks.append((label, content))
            current.clear()

    def split_long_line(line: str, limit: int) -> List[str]:
        parts = []
        remaining = line.strip()
        while len(remaining) > limit:
            cut = remaining.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            parts.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            parts.append(remaining)
        return parts

    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            section = heading.group(1).strip()
            continue
        label = section[:max(1, max_chars // 3)]
        line_limit = max(1, max_chars - len(label) - 1)
        for part in split_long_line(line, line_limit):
            candidate = "\n".join(current + [part])
            if len(label) + 1 + len(candidate) > max_chars:
                flush()
            current.append(part)
    flush()
    return chunks


def _load_plugin_config() -> Dict[str, Any]:
    """Read memory.ollama_memory block from config.yaml."""
    try:
        from hermes_cli.config import load_config_readonly
        block = load_config_readonly().get("memory", {}).get("ollama_memory", {})
    except Exception:
        block = None
    return dict(block) if isinstance(block, dict) else {}


def _schema(name: str, description: str, properties: Dict | None = None, required: Tuple = ()) -> Dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": list(required)}}


def _p(description: str, type_: str = "string", **extra) -> Dict:
    return {"type": type_, **extra, "description": description}


_SCHEMAS = (
    _schema("ollama_search", "Semantic search across stored memories using Ollama embeddings. Returns ranked results with relevance scores.",
            {"query": _p("What to search for."), "top_k": _p("Max results (default: 8, max: 20).", "integer")}, ("query",)),
    _schema("ollama_remember", "Persist an explicit fact, preference, or decision to long-term memory via Ollama embeddings.",
            {"content": _p("The fact to remember."),
             "memory_type": _p("Category (default: factual).", enum=["factual", "preference", "goal", "instruction", "event", "opinion"]),
             "importance": _p("Importance 0-1 (default: 0.7).", "number")}, ("content",)),
    _schema("ollama_forget", "Delete a specific memory by ID.", {"memory_id": _p("Memory ID to delete.")}, ("memory_id",)),
    _schema("ollama_status", "Check Ollama memory provider status — endpoint, model, collection stats."),
    _schema("ollama_reindex", "Recompute embeddings for all memories (use if model/endpoint changed)."),
)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class _OllamaClient:
    """Thin HTTP client for Ollama embeddings API."""

    def __init__(self, endpoint: str, model: str, keep_alive: str = "10m", cache_size: int = 128):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.cache_size = max(0, int(cache_size))
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._prefetch_gate = threading.BoundedSemaphore(1)

    def embed(self, text: str, timeout: float = 30) -> List[float]:
        """Get one embedding, using the bounded in-memory cache when possible."""
        return self.embed_batch([text], timeout=timeout)[0]

    def embed_bounded(self, text: str, timeout: float) -> List[float]:
        """Wait no longer than `timeout` wall-clock seconds for a prefetch embedding."""
        if not self._prefetch_gate.acquire(blocking=False):
            raise TimeoutError("another prefetch embedding is still in flight")
        result: Dict[str, Any] = {}
        completed = threading.Event()

        def _run() -> None:
            try:
                result["embedding"] = self.embed(text, timeout=30)
            except Exception as exc:
                result["error"] = exc
            finally:
                self._prefetch_gate.release()
                completed.set()

        worker = threading.Thread(target=_run, name="ollama-memory-prefetch-embedding", daemon=True)
        try:
            worker.start()
        except Exception:
            self._prefetch_gate.release()
            raise
        if not completed.wait(max(0.0, float(timeout))):
            raise TimeoutError("prefetch embedding exceeded its wall-clock budget")
        if "error" in result:
            raise result["error"]
        return result["embedding"]

    def embed_batch(self, texts: List[str], timeout: float = 60) -> List[List[float]]:
        """Get embeddings in one request and cache recent results by content digest."""
        if not texts:
            return []
        results: List[Optional[List[float]]] = [None] * len(texts)
        pending: Dict[str, Tuple[str, List[int]]] = {}
        with self._cache_lock:
            for index, text in enumerate(texts):
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                cached = self._embedding_cache.get(key)
                if cached is not None:
                    self._embedding_cache.move_to_end(key)
                    results[index] = list(cached)
                else:
                    if key not in pending:
                        pending[key] = (text, [])
                    pending[key][1].append(index)

        if pending:
            pending_items = list(pending.items())
            for offset in range(0, len(pending_items), _EMBED_BATCH_SIZE):
                batch_items = pending_items[offset:offset + _EMBED_BATCH_SIZE]
                missing = [item[1][0] for item in batch_items]
                payload = {"model": self.model, "input": missing, "keep_alive": self.keep_alive}
                try:
                    response = self._session.post(
                        f"{self.endpoint}/api/embed", json=payload, timeout=timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings = data.get("embeddings")
                    if not isinstance(embeddings, list) or len(embeddings) != len(missing):
                        raise RuntimeError("Ollama returned an invalid embedding batch")
                except requests.HTTPError as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status not in {404, 405}:
                        raise
                    embeddings = []
                    for text in missing:
                        legacy = self._session.post(
                            f"{self.endpoint}/api/embeddings",
                            json={"model": self.model, "prompt": text}, timeout=timeout,
                        )
                        legacy.raise_for_status()
                        vector = legacy.json().get("embedding")
                        if not vector:
                            raise RuntimeError("Ollama returned no embedding")
                        embeddings.append(vector)

                with self._cache_lock:
                    for (key, (text, indices)), vector in zip(batch_items, embeddings):
                        if not vector:
                            raise RuntimeError("Ollama returned an empty embedding")
                        vector = [float(value) for value in vector]
                        if self.cache_size:
                            self._embedding_cache[key] = vector
                            self._embedding_cache.move_to_end(key)
                            while len(self._embedding_cache) > self.cache_size:
                                self._embedding_cache.popitem(last=False)
                        for index in indices:
                            results[index] = list(vector)

        if any(vector is None for vector in results):
            raise RuntimeError("Ollama returned an incomplete embedding batch")
        return [vector for vector in results if vector is not None]

    def health_check(self) -> bool:
        try:
            resp = self._session.get(f"{self.endpoint}/api/tags", timeout=5)
            return resp.ok
        except Exception:
            return False


class _MemoryStore:
    """SQLite-backed memory store with vector embeddings."""

    def __init__(self, db_path: Path, collection: str, dim: int = _EMBED_DIM):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", collection):
            raise ValueError("collection must be a SQLite identifier (letters, digits, underscores)")
        self.db_path = db_path
        self.collection = collection
        self.dim = dim
        self._lock = threading.Lock()
        self._local = threading.local()
        self._fts_available = False
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def _fts_table(self) -> str:
        return f"memories_fts_{self.collection}"

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS memories_{self.collection} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    memory_type TEXT DEFAULT 'factual',
                    importance REAL DEFAULT 0.7,
                    metadata JSON DEFAULT '{{}}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_{self.collection}_type
                ON memories_{self.collection}(memory_type)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_{self.collection}_created
                ON memories_{self.collection}(created_at)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ollama_memory_imports (
                    collection TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    imported_at TEXT NOT NULL,
                    PRIMARY KEY (collection, source_file, content_hash)
                )
            """)
            try:
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._fts_table} "
                    "USING fts5(content, memory_id UNINDEXED, tokenize='unicode61')"
                )
                conn.execute(
                    f"DELETE FROM {self._fts_table} "
                    f"WHERE rowid NOT IN (SELECT id FROM memories_{self.collection})"
                )
                conn.execute(
                    f"""INSERT INTO {self._fts_table}(rowid, content, memory_id)
                        SELECT m.id, m.content, CAST(m.id AS TEXT)
                        FROM memories_{self.collection} AS m
                        LEFT JOIN {self._fts_table} AS f ON f.rowid = m.id
                        WHERE f.rowid IS NULL"""
                )
                self._fts_available = True
            except sqlite3.OperationalError as exc:
                logger.warning("SQLite FTS5 unavailable; lexical fallback will be used (%s)",
                               type(exc).__name__)
            conn.commit()

    def add(self, content: str, embedding: List[float], memory_type: str = "factual",
            importance: float = 0.7, metadata: Dict | None = None) -> int:
        now = datetime.now(UTC).isoformat()
        emb_bytes = json.dumps(embedding).encode()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        conn = self._get_conn()
        cur = conn.execute(
            f"""INSERT INTO memories_{self.collection}
                (content, embedding, memory_type, importance, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (content, emb_bytes, memory_type, importance, meta_json, now, now))
        if self._fts_available:
            conn.execute(
                f"INSERT INTO {self._fts_table}(rowid, content, memory_id) VALUES (?, ?, ?)",
                (cur.lastrowid, content, str(cur.lastrowid)),
            )
        conn.commit()
        return cur.lastrowid

    def source_imported(self, source_file: str, content_hash: str) -> bool:
        conn = self._get_conn()
        with self._lock:
            row = conn.execute(
                "SELECT 1 FROM ollama_memory_imports "
                "WHERE collection = ? AND source_file = ? AND content_hash = ? LIMIT 1",
                (self.collection, source_file, content_hash),
            ).fetchone()
        return row is not None

    def import_source(self, source_file: str, content_hash: str, chunks: List[str],
                      embeddings: List[List[float]], memory_type: str, importance: float,
                      sections: Optional[List[str]] = None, source_sha256: Optional[str] = None,
                      import_version: int = _PROFILE_IMPORT_VERSION) -> bool:
        """Atomically append a source snapshot and its import marker; never replace prior data."""
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("source chunks and embeddings must be non-empty and have equal lengths")
        if sections is not None and len(sections) != len(chunks):
            raise ValueError("source sections must match the chunk count")
        source_sections = sections or ["General"] * len(chunks)
        now = datetime.now(UTC).isoformat()
        conn = self._get_conn()
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT 1 FROM ollama_memory_imports "
                    "WHERE collection = ? AND source_file = ? AND content_hash = ? LIMIT 1",
                    (self.collection, source_file, content_hash),
                ).fetchone()
                if existing:
                    conn.rollback()
                    return False
                for index, (content, embedding, section) in enumerate(
                        zip(chunks, embeddings, source_sections), start=1):
                    metadata = json.dumps({
                        "source_file": source_file,
                        "source_sha256": source_sha256 or content_hash,
                        "import_version": import_version,
                        "section": section,
                        "chunk_index": index,
                        "chunk_count": len(chunks),
                    }, ensure_ascii=False)
                    inserted = conn.execute(
                        f"""INSERT INTO memories_{self.collection}
                            (content, embedding, memory_type, importance, metadata, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (content, json.dumps(embedding).encode(), memory_type, importance,
                         metadata, now, now),
                    )
                    if self._fts_available:
                        conn.execute(
                            f"INSERT INTO {self._fts_table}(rowid, content, memory_id) VALUES (?, ?, ?)",
                            (inserted.lastrowid, content, str(inserted.lastrowid)),
                        )
                conn.execute(
                    "INSERT INTO ollama_memory_imports "
                    "(collection, source_file, content_hash, chunk_count, imported_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (self.collection, source_file, content_hash, len(chunks), now),
                )
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise

    def search(self, query_embedding: List[float], top_k: int = 8,
               threshold: float = 0.25, memory_type: Optional[str] = None) -> List[Dict]:
        conn = self._get_conn()
        sql = f"""SELECT id, content, embedding, memory_type, importance, metadata, created_at
                  FROM memories_{self.collection}"""
        params = []
        if memory_type:
            sql += " WHERE memory_type = ?"
            params.append(memory_type)
        sql += " ORDER BY created_at DESC LIMIT 5000"  # cap scan for performance
        rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            stored_emb = json.loads(row["embedding"].decode())
            sim = _cosine_similarity(query_embedding, stored_emb)
            if sim >= threshold:
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "memory_type": row["memory_type"],
                    "importance": row["importance"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                    "created_at": row["created_at"],
                    "similarity": round(sim, 4),
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def search_lexical(self, query: str, top_k: int = 8) -> List[Dict]:
        """Retrieve exact terms with FTS5; fall back to a bounded local scan if unavailable."""
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)[:32]
        if not terms or top_k <= 0:
            return []
        conn = self._get_conn()
        if self._fts_available:
            match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
            with self._lock:
                rows = conn.execute(
                    f"""SELECT m.id, m.content, m.memory_type, m.importance, m.metadata,
                               m.created_at, bm25({self._fts_table}) AS lexical_score
                        FROM {self._fts_table}
                        JOIN memories_{self.collection} AS m ON m.id = {self._fts_table}.rowid
                        WHERE {self._fts_table} MATCH ?
                        ORDER BY lexical_score ASC LIMIT ?""",
                    (match_query, top_k),
                ).fetchall()
            return [{
                "id": row["id"], "content": row["content"],
                "memory_type": row["memory_type"], "importance": row["importance"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "created_at": row["created_at"], "similarity": 0.0,
                "lexical_score": row["lexical_score"], "lexical_rank": rank,
            } for rank, row in enumerate(rows, start=1)]

        rows = conn.execute(
            f"""SELECT id, content, memory_type, importance, metadata, created_at
                FROM memories_{self.collection} ORDER BY created_at DESC LIMIT 5000"""
        ).fetchall()
        ranked = []
        folded_terms = [term.casefold() for term in terms]
        for row in rows:
            content = row["content"]
            folded = content.casefold()
            hits = sum(term in folded for term in folded_terms)
            if not hits:
                continue
            ranked.append((hits, row))
        ranked.sort(key=lambda item: (item[0], item[1]["created_at"]), reverse=True)
        return [{
            "id": row["id"], "content": row["content"],
            "memory_type": row["memory_type"], "importance": row["importance"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "created_at": row["created_at"], "similarity": 0.0,
            "lexical_rank": rank,
        } for rank, (_, row) in enumerate(ranked[:top_k], start=1)]

    def search_hybrid(self, query_embedding: Optional[List[float]], query: str,
                      top_k: int = 8, threshold: float = 0.25) -> List[Dict]:
        """Fuse vector similarity and exact lexical ranks using weighted reciprocal rank fusion."""
        if top_k <= 0:
            return []
        limit = max(top_k * 3, top_k)
        vector_rows = self.search(query_embedding, top_k=limit, threshold=threshold) if query_embedding else []
        lexical_rows = self.search_lexical(query, top_k=limit)
        merged: Dict[int, Dict] = {}
        for rank, row in enumerate(vector_rows, start=1):
            merged[row["id"]] = {**row, "vector_rank": rank, "lexical_rank": None}
        for rank, row in enumerate(lexical_rows, start=1):
            target = merged.setdefault(row["id"], {**row, "vector_rank": None, "lexical_rank": None})
            target["lexical_rank"] = rank
            target["lexical_score"] = row.get("lexical_score")
            if target.get("similarity") is None:
                target["similarity"] = 0.0
        for row in merged.values():
            score = 0.0
            if row["vector_rank"] is not None:
                score += 1.0 / (60 + row["vector_rank"])
            if row["lexical_rank"] is not None:
                score += 2.0 / (60 + row["lexical_rank"])
            row["hybrid_score"] = score
        ranked = sorted(merged.values(), key=lambda row: (
            row["hybrid_score"], row.get("similarity", 0.0), row.get("importance", 0.0)
        ), reverse=True)
        seen_content = set()
        unique = []
        for row in ranked:
            canonical = " ".join(row["content"].split()).casefold()
            if canonical in seen_content:
                continue
            seen_content.add(canonical)
            unique.append(row)
            if len(unique) >= top_k:
                break
        return unique

    def delete(self, memory_id: int) -> bool:
        conn = self._get_conn()
        with self._lock:
            if self._fts_available:
                conn.execute(f"DELETE FROM {self._fts_table} WHERE rowid = ?", (memory_id,))
            cur = conn.execute(f"DELETE FROM memories_{self.collection} WHERE id = ?", (memory_id,))
            conn.commit()
        return cur.rowcount > 0

    def get(self, memory_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT id, content, embedding, memory_type, importance, metadata, created_at "
            f"FROM memories_{self.collection} WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "content": row["content"],
            "embedding": json.loads(row["embedding"].decode()),
            "memory_type": row["memory_type"],
            "importance": row["importance"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "created_at": row["created_at"],
        }

    def stats(self) -> Dict:
        conn = self._get_conn()
        row = conn.execute(f"SELECT COUNT(*) as cnt FROM memories_{self.collection}").fetchone()
        imported = conn.execute(
            "SELECT COUNT(DISTINCT source_file) AS cnt FROM ollama_memory_imports WHERE collection = ?",
            (self.collection,),
        ).fetchone()
        return {"collection": self.collection, "count": row["cnt"], "dimension": self.dim,
                "imported_sources": imported["cnt"]}

    def reindex(self, client: _OllamaClient) -> Tuple[int, int]:
        """Recompute all embeddings using current client. Returns (updated, failed)."""
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT id, content FROM memories_{self.collection}").fetchall()
        updated = failed = 0
        for row in rows:
            try:
                emb = client.embed(row["content"])
                emb_bytes = json.dumps(emb).encode()
                conn.execute(
                    f"UPDATE memories_{self.collection} SET embedding = ?, updated_at = ? WHERE id = ?",
                    (emb_bytes, datetime.now(UTC).isoformat(), row["id"]))
                updated += 1
            except Exception as e:
                logger.warning("Reindex failed for memory %d: %s", row["id"], e)
                failed += 1
        conn.commit()
        return updated, failed


def _compact(s: str) -> str:
    """Normalize whitespace without silently dropping the end of a memory chunk."""
    return " ".join(str(s or "").split())


def _query_excerpt(text: str, query: str, max_chars: int) -> str:
    """Keep the densest matching window when a full memory chunk exceeds the budget."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    terms = [term.casefold() for term in re.findall(r"[\w.+-]+", query, flags=re.UNICODE)
             if len(term) > 1]
    folded = text.casefold()
    matches = []
    for term in terms:
        start = 0
        while len(matches) < 200:
            found = folded.find(term, start)
            if found < 0:
                break
            matches.append((term, found))
            start = found + max(1, len(term))
    if not matches:
        return text[:max_chars]
    max_start = max(0, len(text) - max_chars)
    starts = {max(0, min(max_start, pos - max_chars // 3)) for _, pos in matches}

    def window_score(point: int) -> Tuple[int, int, int]:
        matched_terms = {term for term, pos in matches if point <= pos < point + max_chars}
        return sum(len(term) for term in matched_terms), len(matched_terms), point

    start = max(starts, key=window_score)
    excerpt = text[start:start + max_chars]
    if start:
        excerpt = "…" + excerpt[1:]
    if start + max_chars < len(text):
        excerpt = excerpt[:-1] + "…"
    return excerpt


def _build_context_block(results: List[Dict], max_tokens: int, query: str = "") -> str:
    """Format search results into a context block for the LLM."""
    if not results:
        return ""
    header = "[Ollama Memory Context]"
    lines = [header]
    char_budget = max(0, max_tokens) * 4
    used = len(header)
    for r in results:
        prefix = f"- [{r['memory_type']}] "
        if r.get("vector_rank") is None and r.get("lexical_rank") is not None:
            suffix = " (exact lexical match)"
        else:
            suffix = f" (sim: {r.get('similarity', 0.0):.2f})"
        available = char_budget - used - len(prefix) - len(suffix) - 1
        if available <= 0:
            break
        content = _compact(r.get("content", ""))
        if len(content) > available:
            content = _query_excerpt(content, query, available)
        if not content:
            continue
        entry = prefix + content + suffix
        if used + len(entry) + 1 > char_budget:
            break
        lines.append(entry)
        used += len(entry) + 1
    return "\n".join(lines)


class OllamaMemoryProvider(MemoryProvider):
    """Ollama-backed local semantic memory provider."""

    def __init__(self):
        self._client: Optional[_OllamaClient] = None
        self._store: Optional[_MemoryStore] = None
        self._config: Dict[str, Any] = {}
        self._endpoint = _DEFAULT_ENDPOINT
        self._model = _DEFAULT_MODEL
        self._collection = _DEFAULT_COLLECTION
        self._top_k = _DEFAULT_TOP_K
        self._threshold = _DEFAULT_SIMILARITY_THRESHOLD
        self._max_context_tokens = _DEFAULT_MAX_CONTEXT_TOKENS
        self._keep_alive = _DEFAULT_KEEP_ALIVE
        self._prefetch_embedding_timeout = _DEFAULT_PREFETCH_EMBEDDING_TIMEOUT
        self._auto_embed_writes = True
        self._initialized = False

    @property
    def name(self) -> str:
        return "ollama_memory"

    def is_available(self) -> bool:
        # Hermes calls this before initialize(); avoid a network call here.
        return bool(self._endpoint and self._model)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "endpoint", "description": "Ollama API base URL", "default": _DEFAULT_ENDPOINT},
            {"key": "model", "description": "Embedding model name", "default": _DEFAULT_MODEL},
            {"key": "collection", "description": "Logical collection name (table suffix)", "default": _DEFAULT_COLLECTION},
            {"key": "top_k", "description": "Default search results", "type": "integer", "default": _DEFAULT_TOP_K},
            {"key": "similarity_threshold", "description": "Cosine similarity cutoff 0-1", "type": "number", "default": _DEFAULT_SIMILARITY_THRESHOLD},
            {"key": "max_context_tokens", "description": "Max tokens in prefetch context block", "type": "integer", "default": _DEFAULT_MAX_CONTEXT_TOKENS},
            {"key": "keep_alive", "description": "Keep the embedding model loaded for this duration after a request", "default": _DEFAULT_KEEP_ALIVE},
            {"key": "prefetch_embedding_timeout", "description": "Seconds to wait for embeddings before using lexical-only fallback", "type": "number", "default": _DEFAULT_PREFETCH_EMBEDDING_TIMEOUT},
            {"key": "auto_embed_writes", "description": "Mirror built-in memory.add to Ollama store", "type": "boolean", "default": True},
            {"key": "import_profile_files", "description": "Opt in to embedding MEMORY.md, USER.md and SOUL.md at the configured endpoint", "type": "boolean", "default": False},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_plugin_config()
        self._endpoint = self._config.get("endpoint", _DEFAULT_ENDPOINT)
        self._model = self._config.get("model", _DEFAULT_MODEL)
        self._collection = self._config.get("collection", _DEFAULT_COLLECTION)
        self._top_k = int(self._config.get("top_k", _DEFAULT_TOP_K))
        self._threshold = float(self._config.get("similarity_threshold", _DEFAULT_SIMILARITY_THRESHOLD))
        self._max_context_tokens = int(self._config.get("max_context_tokens", _DEFAULT_MAX_CONTEXT_TOKENS))
        self._keep_alive = str(self._config.get("keep_alive", _DEFAULT_KEEP_ALIVE))
        try:
            configured_timeout = float(self._config.get(
                "prefetch_embedding_timeout", _DEFAULT_PREFETCH_EMBEDDING_TIMEOUT
            ))
        except (TypeError, ValueError):
            configured_timeout = _DEFAULT_PREFETCH_EMBEDDING_TIMEOUT
        self._prefetch_embedding_timeout = min(6.0, max(0.5, configured_timeout))
        self._auto_embed_writes = bool(self._config.get("auto_embed_writes", True))

        # Bind to the profile the manager names (multiplexing contract)
        from hermes_constants import get_hermes_home
        home = Path(kwargs["hermes_home"]) if kwargs.get("hermes_home") else get_hermes_home()

        self._client = _OllamaClient(self._endpoint, self._model, keep_alive=self._keep_alive)
        self._store = _MemoryStore(home / "ollama_memory.db", self._collection)
        self._initialized = True
        self._import_profile_files(home)
        spawn_context_thread(self._warm_up, name="ollama-memory-warmup").start()
        logger.info("OllamaMemoryProvider initialized: endpoint=%s model=%s collection=%s",
                    self._endpoint, self._model, self._collection)

    def _warm_up(self) -> None:
        """Prime Ollama in the background with a synthetic, non-user prompt."""
        client = self._client
        if client is None:
            return
        try:
            client.embed("Hermes local memory embedding warm-up", timeout=self._prefetch_embedding_timeout)
        except Exception as exc:
            logger.debug("Ollama embedding warm-up failed (%s)", type(exc).__name__)

    def _import_profile_files(self, home: Path) -> None:
        """Append opted-in profile Markdown snapshots without escaping the active profile."""
        enabled = self._config.get("import_profile_files", False)
        if not (enabled is True or isinstance(enabled, str)
                and enabled.strip().lower() in {"1", "true", "yes", "on"}):
            return
        client, store = self._client, self._store
        if client is None or store is None:
            return
        home_root = home.resolve()
        for candidates, memory_type, importance in _PROFILE_MEMORY_SOURCES:
            source_path = None
            source_file = None
            for candidate in candidates:
                candidate_path = home / candidate
                try:
                    resolved_path = candidate_path.resolve(strict=True)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    logger.warning("Could not resolve profile source %s (%s)",
                                   candidate, type(exc).__name__)
                    continue
                try:
                    resolved_path.relative_to(home_root)
                except ValueError:
                    logger.warning("Refusing profile source outside active home: %s", candidate)
                    continue
                if resolved_path.is_file():
                    source_path = resolved_path
                    source_file = candidate
                    break
            if source_path is None or source_file is None:
                continue
            try:
                text = source_path.read_text(encoding="utf-8", errors="replace").strip()
                chunk_rows = _split_for_embedding(text)
                if not chunk_rows:
                    continue
                sections = [section for section, _ in chunk_rows]
                chunks = [content for _, content in chunk_rows]
                source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                snapshot_hash = hashlib.sha256(
                    f"profile-import-v{_PROFILE_IMPORT_VERSION}:{source_sha256}".encode("ascii")
                ).hexdigest()
                if store.source_imported(source_file, snapshot_hash):
                    continue
                embeddings = client.embed_batch(chunks)
                if store.import_source(
                        source_file, snapshot_hash, chunks, embeddings, memory_type, importance,
                        sections=sections, source_sha256=source_sha256,
                        import_version=_PROFILE_IMPORT_VERSION):
                    logger.info("Imported %d chunks from profile memory source %s",
                                len(chunks), source_file)
            except Exception as exc:
                # Never log source contents; files can contain private user and assistant data.
                logger.warning("Could not import profile memory source %s (%s)",
                               source_file, type(exc).__name__)

    def system_prompt_block(self) -> str:
        if not self._initialized:
            return ""
        return (f"# Ollama Memory\nActive. Endpoint: {self._endpoint}, Model: {self._model}, "
                f"Collection: {self._collection}.\n"
                "Use ollama_search to find memories, ollama_remember to store facts, "
                "ollama_status to check state, ollama_reindex to rebuild embeddings.")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Hybrid turn-start search with a bounded embedding wait and lexical fallback."""
        if not self._initialized or not query or len(query.strip()) < 3:
            return ""
        client, store = self._client, self._store
        if client is None or store is None:
            return ""
        started = time.monotonic()
        try:
            q_emb = client.embed_bounded(query, timeout=self._prefetch_embedding_timeout)
        except Exception as exc:
            elapsed = time.monotonic() - started
            logger.info("Ollama prefetch embedding failed; using lexical fallback (%s, %.2fs)",
                        type(exc).__name__, elapsed)
            q_emb = None
        try:
            results = store.search_hybrid(
                q_emb, query, top_k=self._top_k, threshold=self._threshold
            )
        except Exception as exc:
            logger.warning("Ollama hybrid prefetch failed (%s)", type(exc).__name__)
            return ""
        elapsed = time.monotonic() - started
        logger.debug("Ollama hybrid prefetch completed in %.2fs with %d results", elapsed, len(results))
        return _build_context_block(results, self._max_context_tokens, query=query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch() runs synchronously at turn start."""
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Embed and store the conversation turn."""
        if not self._initialized or not self._auto_embed_writes:
            return
        if not user_content or len(user_content.strip()) < 10:
            return
        try:
            text = f"User: {user_content[:2000]}\nAssistant: {assistant_content[:2000]}"
            emb = self._client.embed(text)
            self._store.add(text, emb, memory_type="conversation", importance=0.5,
                            metadata={"session_id": session_id})
        except Exception as e:
            logger.debug("Ollama sync_turn failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to Ollama store."""
        if not self._initialized or not self._auto_embed_writes:
            return
        if action not in {"add", "replace"} or not content:
            return
        try:
            mem_type = "preference" if target == "user" else "factual"
            emb = self._client.embed(content)
            self._store.add(content, emb, memory_type=mem_type, importance=0.8,
                            metadata={"source": f"memory.{target}", "action": action})
        except Exception as e:
            logger.debug("Ollama memory mirror failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict, **kwargs) -> str:
        if not self._initialized:
            return tool_error("Ollama memory not initialized")
        try:
            if tool_name == "ollama_search":
                return self._tool_search(args)
            elif tool_name == "ollama_remember":
                return self._tool_remember(args)
            elif tool_name == "ollama_forget":
                return self._tool_forget(args)
            elif tool_name == "ollama_status":
                return self._tool_status()
            elif tool_name == "ollama_reindex":
                return self._tool_reindex()
            else:
                return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            return tool_error(str(e))

    def _tool_search(self, args: Dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        top_k = min(max(1, int(args.get("top_k", self._top_k))), 20)
        client, store = self._client, self._store
        if client is None or store is None:
            return tool_error("Ollama memory is not initialized")
        try:
            q_emb = client.embed(query)
        except Exception as exc:
            logger.info("Ollama explicit search using lexical-only fallback (%s)", type(exc).__name__)
            q_emb = None
        results = store.search_hybrid(
            q_emb, query, top_k=top_k, threshold=self._threshold
        )
        return json.dumps({"query": query, "results": results, "count": len(results)}, ensure_ascii=False)

    def _tool_remember(self, args: Dict) -> str:
        content = (args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        mem_type = args.get("memory_type", "factual")
        importance = float(args.get("importance", 0.7))
        emb = self._client.embed(content)
        mid = self._store.add(content, emb, memory_type=mem_type, importance=importance)
        return json.dumps({"status": "stored", "id": mid, "content": content[:100]}, ensure_ascii=False)

    def _tool_forget(self, args: Dict) -> str:
        mid = args.get("memory_id")
        if mid is None:
            return tool_error("memory_id is required")
        ok = self._store.delete(int(mid))
        return json.dumps({"status": "deleted" if ok else "not_found", "id": mid}, ensure_ascii=False)

    def _tool_status(self) -> str:
        stats = self._store.stats() if self._store else {"error": "not initialized"}
        stats.update({
            "endpoint": self._endpoint,
            "model": self._model,
            "initialized": self._initialized,
            "available": self.is_available(),
        })
        return json.dumps(stats, ensure_ascii=False)

    def _tool_reindex(self) -> str:
        if not self._client or not self._store:
            return tool_error("not initialized")
        updated, failed = self._store.reindex(self._client)
        return json.dumps({"status": "reindexed", "updated": updated, "failed": failed}, ensure_ascii=False)

    def shutdown(self) -> None:
        if self._store:
            # close thread-local connections
            if hasattr(self._store._local, "conn"):
                with suppress(Exception):
                    self._store._local.conn.close()
        self._initialized = False


def register(ctx) -> None:
    """Register OllamaMemoryProvider as a memory provider plugin."""
    ctx.register_memory_provider(OllamaMemoryProvider())