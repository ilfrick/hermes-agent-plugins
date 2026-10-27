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

import json
import logging
import os
import sqlite3
import threading
import time
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
_EMBED_DIM = 2560  # qwen3-embedding:4b dimension


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

    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text. Returns list of floats."""
        payload = {"model": self.model, "prompt": text}
        resp = self._session.post(f"{self.endpoint}/api/embeddings", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(f"Ollama returned no embedding: {data}")
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (sequential, for simplicity)."""
        return [self.embed(t) for t in texts]

    def health_check(self) -> bool:
        try:
            resp = self._session.get(f"{self.endpoint}/api/tags", timeout=5)
            return resp.ok
        except Exception:
            return False


class _MemoryStore:
    """SQLite-backed memory store with vector embeddings."""

    def __init__(self, db_path: Path, collection: str, dim: int = _EMBED_DIM):
        self.db_path = db_path
        self.collection = collection
        self.dim = dim
        self._lock = threading.Lock()
        self._local = threading.local()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

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
        conn.commit()
        return cur.lastrowid

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

    def delete(self, memory_id: int) -> bool:
        conn = self._get_conn()
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
        return {"collection": self.collection, "count": row["cnt"], "dimension": self.dim}

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


def _compact(s: str, max_len: int = 320) -> str:
    return " ".join(str(s or "").split())[:max_len]


def _build_context_block(results: List[Dict], max_tokens: int) -> str:
    """Format search results into a context block for the LLM."""
    if not results:
        return ""
    lines = ["[Ollama Memory Context]"]
    # rough token estimate: 1 token ~ 4 chars
    char_budget = max_tokens * 4
    used = 0
    for r in results:
        entry = f"- [{r['memory_type']}] {_compact(r['content'])} (sim: {r['similarity']:.2f})"
        if used + len(entry) > char_budget:
            break
        lines.append(entry)
        used += len(entry)
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
            {"key": "auto_embed_writes", "description": "Mirror built-in memory.add to Ollama store", "type": "boolean", "default": True},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_plugin_config()
        self._endpoint = self._config.get("endpoint", _DEFAULT_ENDPOINT)
        self._model = self._config.get("model", _DEFAULT_MODEL)
        self._collection = self._config.get("collection", _DEFAULT_COLLECTION)
        self._top_k = int(self._config.get("top_k", _DEFAULT_TOP_K))
        self._threshold = float(self._config.get("similarity_threshold", _DEFAULT_SIMILARITY_THRESHOLD))
        self._max_context_tokens = int(self._config.get("max_context_tokens", _DEFAULT_MAX_CONTEXT_TOKENS))
        self._auto_embed_writes = bool(self._config.get("auto_embed_writes", True))

        # Bind to the profile the manager names (multiplexing contract)
        from hermes_constants import get_hermes_home
        home = Path(kwargs["hermes_home"]) if kwargs.get("hermes_home") else get_hermes_home()

        self._client = _OllamaClient(self._endpoint, self._model)
        self._store = _MemoryStore(home / "ollama_memory.db", self._collection)
        self._initialized = True
        logger.info("OllamaMemoryProvider initialized: endpoint=%s model=%s collection=%s",
                    self._endpoint, self._model, self._collection)

    def system_prompt_block(self) -> str:
        if not self._initialized:
            return ""
        return (f"# Ollama Memory\nActive. Endpoint: {self._endpoint}, Model: {self._model}, "
                f"Collection: {self._collection}.\n"
                "Use ollama_search to find memories, ollama_remember to store facts, "
                "ollama_status to check state, ollama_reindex to rebuild embeddings.")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Synchronous search at turn start so context is ready before first LLM call."""
        if not self._initialized or not query or len(query.strip()) < 3:
            return ""
        try:
            q_emb = self._client.embed(query)
            results = self._store.search(q_emb, top_k=self._top_k, threshold=self._threshold)
            return _build_context_block(results, self._max_context_tokens)
        except Exception as e:
            logger.debug("Ollama prefetch failed: %s", e)
            return ""

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
        top_k = min(int(args.get("top_k", self._top_k)), 20)
        q_emb = self._client.embed(query)
        results = self._store.search(q_emb, top_k=top_k, threshold=self._threshold)
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