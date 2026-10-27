import importlib.util
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PLUGIN_FILE = Path(__file__).resolve().parents[1] / "__init__.py"

# Stub only the host interfaces so the standalone provider can be tested without
# loading a Hermes gateway or modifying the user's active profile.
agent = types.ModuleType("agent")
memory_api = types.ModuleType("agent.memory_provider")
setattr(memory_api, "MemoryProvider", type("MemoryProvider", (), {}))


class ImmediateThread:
    def __init__(self, target, args=(), kwargs=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


def _spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None):
    return ImmediateThread(target, args=args, kwargs=kwargs)


setattr(memory_api, "spawn_context_thread", _spawn_context_thread)
setattr(agent, "memory_provider", memory_api)
tools = types.ModuleType("tools")
registry = types.ModuleType("tools.registry")
setattr(registry, "tool_error", lambda message: json.dumps({"error": message}))
setattr(tools, "registry", registry)
spec = importlib.util.spec_from_file_location("ollama_memory_plugin_test", PLUGIN_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load provider from {PLUGIN_FILE}")
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
with patch.dict(sys.modules, {
    "agent": agent,
    "agent.memory_provider": memory_api,
    "tools": tools,
    "tools.registry": registry,
}):
    spec.loader.exec_module(plugin)


class FakeOllamaClient:
    def __init__(self, endpoint, model, keep_alive="10m", cache_size=128):
        self.endpoint = endpoint
        self.model = model
        self.keep_alive = keep_alive

    def embed(self, text, timeout=30):
        return [1.0, 0.0]

    def embed_bounded(self, text, timeout):
        return self.embed(text, timeout=timeout)

    def embed_batch(self, texts, timeout=30):
        return [self.embed(text, timeout=timeout) for text in texts]

    def health_check(self):
        return True


class FailOnceOllamaClient(FakeOllamaClient):
    fail_next_embedding = True

    def embed(self, text, timeout=30):
        if type(self).fail_next_embedding:
            type(self).fail_next_embedding = False
            raise RuntimeError("simulated embedding failure")
        return super().embed(text, timeout=timeout)


class TimeoutOllamaClient(FakeOllamaClient):
    timeouts = []

    def embed(self, text, timeout=30):
        type(self).timeouts.append(timeout)
        raise plugin.requests.Timeout("simulated slow embedding")


class ProfileImportTests(unittest.TestCase):
    def search_all(self, provider):
        return json.loads(provider.handle_tool_call(
            "ollama_search", {"query": "retrieve imported profile content", "top_k": 20}
        ))["results"]

    def test_imports_memory_user_and_soul_idempotently_and_preserves_updates(self):
        source_text = {
            "memories/MEMORY.md": "Stable memory fact: telemetry is stored locally.",
            "memories/USER.md": "User preference: use concise Italian replies.",
            "SOUL.md": "Assistant identity: be candid and evidence-led.",
        }
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(plugin, "_OllamaClient", FakeOllamaClient),
            patch.object(plugin, "_load_plugin_config", return_value={"import_profile_files": True}),
        ):
            home = Path(td)
            for relative_path, text in source_text.items():
                path = home / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")

            first = plugin.OllamaMemoryProvider()
            first.initialize(session_id="test", hermes_home=str(home))
            first_rows = self.search_all(first)
            first_count = len(first_rows)

            self.assertGreaterEqual(first_count, 3)
            self.assertEqual(
                {row["metadata"]["source_file"] for row in first_rows}, set(source_text)
            )
            self.assertTrue(all(row["metadata"]["import_version"] == 2 for row in first_rows))
            self.assertTrue(all(row["metadata"]["section"] for row in first_rows))
            all_content = " ".join(row["content"] for row in first_rows)
            self.assertTrue(all(text in all_content for text in source_text.values()))
            first.shutdown()

            # Reinitialization must not duplicate unchanged source snapshots.
            second = plugin.OllamaMemoryProvider()
            second.initialize(session_id="test-again", hermes_home=str(home))
            unchanged_rows = self.search_all(second)
            self.assertEqual(len(unchanged_rows), first_count)
            second.shutdown()

            # Changed files are appended as a new snapshot; prior records are retained.
            changed_text = "User preference: provide concise Italian replies and clear caveats."
            (home / "memories" / "USER.md").write_text(changed_text, encoding="utf-8")
            third = plugin.OllamaMemoryProvider()
            third.initialize(session_id="updated", hermes_home=str(home))
            updated_rows = self.search_all(third)
            contents = [row["content"] for row in updated_rows]
            self.assertEqual(len(updated_rows), first_count + 1)
            self.assertTrue(any(source_text["memories/USER.md"] in content for content in contents))
            self.assertTrue(any(changed_text in content for content in contents))
            third.shutdown()


    def test_context_keeps_relevant_details_after_character_320(self):
        content = "Profile summary: " + ("x" * 360) + " Exact phone detail: 555-0102"
        block = plugin._build_context_block(
            [{"memory_type": "preference", "content": content, "similarity": 0.9}],
            max_tokens=1200,
        )
        self.assertIn("Exact phone detail: 555-0102", block)

    def test_context_uses_query_centered_excerpt_when_full_chunk_exceeds_budget(self):
        content = "irrelevant opening " + ("unused detail " * 80) + "Unique tail fact: 555-0102" + (" filler " * 80)
        block = plugin._build_context_block(
            [{"memory_type": "factual", "content": content, "similarity": 0.7}],
            max_tokens=120,
            query="Unique tail fact 555-0102",
        )
        self.assertIn("Unique tail fact: 555-0102", block)
        self.assertLessEqual(len(block), 120 * 4 + len("[Ollama Memory Context]"))

    def test_markdown_chunking_is_bounded_and_carries_section_labels(self):
        source = (
            "## Contacts\n- Name: Alice\n- Phone: 555-0102\n\n"
            "## Preferences\n- Language: Italian\n"
        )
        chunks = plugin._split_for_embedding(source, max_chars=80)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(content) <= 80 for _, content in chunks))
        self.assertIn("Contacts", {section for section, _ in chunks})
        self.assertIn("Preferences", {section for section, _ in chunks})
        self.assertTrue(any("555-0102" in content for _, content in chunks))

    def test_embedding_client_batches_and_caches_queries_with_keep_alive(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}

        class Session:
            def __init__(self):
                self.calls = []
                self.headers = {}

            def post(self, url, json, timeout):
                self.calls.append((url, json, timeout))
                return Response()

        client = plugin._OllamaClient("http://ollama", "test-model", keep_alive="10m")
        session = Session()
        client._session = session
        vectors = client.embed_batch(["alpha", "beta"])
        repeated = client.embed("alpha")
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(repeated, vectors[0])
        self.assertEqual(len(session.calls), 1)
        url, payload, timeout = session.calls[0]
        self.assertEqual(url, "http://ollama/api/embed")
        self.assertEqual(payload["input"], ["alpha", "beta"])
        self.assertEqual(payload["keep_alive"], "10m")
        self.assertEqual(timeout, 60)

    def test_embedding_client_falls_back_to_legacy_endpoint(self):
        class Response:
            status_code = 404

            def raise_for_status(self):
                if self.status_code == 404:
                    error = plugin.requests.HTTPError("not found")
                    error.response = self
                    raise error

            def json(self):
                return {"embedding": [0.5, 0.5]}

        class Session:
            headers = {}

            def __init__(self):
                self.urls = []

            def post(self, url, json, timeout):
                self.urls.append(url)
                response = Response()
                if url.endswith("/api/embeddings"):
                    response.status_code = 200
                return response

        client = plugin._OllamaClient("http://ollama", "legacy-model")
        session = Session()
        client._session = session
        self.assertEqual(client.embed("legacy query"), [0.5, 0.5])
        self.assertEqual(session.urls, ["http://ollama/api/embed", "http://ollama/api/embeddings"])

    def test_embedding_client_enforces_wall_clock_timeout(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": [[1.0, 0.0]]}

        class SlowSession:
            headers = {}

            def post(self, url, json, timeout):
                time.sleep(0.2)
                return Response()

        client = plugin._OllamaClient("http://ollama", "slow-model")
        client._session = SlowSession()
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            client.embed_bounded("slow query", timeout=0.03)
        self.assertLess(time.monotonic() - started, 0.15)
        self.assertTrue(client._prefetch_gate.acquire(timeout=1))
        client._prefetch_gate.release()

    def test_fts_reconciles_partially_missing_index_rows(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "memory.db"
            store = plugin._MemoryStore(db_path, "test")
            store.add("first reconciliation token", [1.0, 0.0])
            second_id = store.add("second reconciliation onlymarker", [0.0, 1.0])
            conn = store._get_conn()
            conn.execute(f"DELETE FROM {store._fts_table} WHERE rowid = ?", (second_id,))
            conn.commit()
            conn.close()

            repaired = plugin._MemoryStore(db_path, "test")
            try:
                rows = repaired.search_lexical("onlymarker", top_k=5)
                self.assertEqual([row["id"] for row in rows], [second_id])
                self.assertIn(1, [row["id"] for row in repaired.search_lexical("first", top_k=5)])
            finally:
                repaired._get_conn().close()

    def test_fts_index_backfills_existing_rows_and_tracks_deletes(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "memory.db"
            old_store = plugin._MemoryStore(db_path, "test")
            old_store._fts_available = False
            memory_id = old_store.add("Unique legacy lookup token", [1.0, 0.0])
            old_store._get_conn().close()

            upgraded_store = plugin._MemoryStore(db_path, "test")
            try:
                rows = upgraded_store.search_lexical("legacy lookup token", top_k=5)
                self.assertEqual([row["id"] for row in rows], [memory_id])
                self.assertTrue(upgraded_store.delete(memory_id))
                self.assertEqual(upgraded_store.search_lexical("legacy lookup token", top_k=5), [])
            finally:
                upgraded_store._get_conn().close()

    def test_hybrid_search_includes_exact_match_below_vector_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            store = plugin._MemoryStore(Path(td) / "memory.db", "test")
            try:
                semantic_id = store.add("A semantically related but unrelated note", [1.0, 0.0])
                exact_id = store.add(
                    "Contact: Elisabetta, WhatsApp phone +39 347 172 2968", [0.0, 1.0]
                )
                results = store.search_hybrid(
                    [1.0, 0.0], "Elisabetta WhatsApp +39 347 172 2968",
                    top_k=5, threshold=0.9,
                )
                result_ids = [row["id"] for row in results]
                self.assertIn(exact_id, result_ids)
                self.assertIn(semantic_id, result_ids)
                self.assertLess(result_ids.index(exact_id), result_ids.index(semantic_id))
            finally:
                store._get_conn().close()

    def test_importer_version_rechunks_legacy_profile_snapshots_append_only(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(plugin, "_OllamaClient", FakeOllamaClient),
            patch.object(plugin, "_load_plugin_config", return_value={"import_profile_files": True}),
        ):
            home = Path(td)
            text = "## Contacts\n" + ("Alice phone detail 555-0102. " * 35)
            (home / "SOUL.md").write_text(text, encoding="utf-8")
            legacy_store = plugin._MemoryStore(home / "ollama_memory.db", "hermes_memory")
            legacy_hash = plugin.hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.assertTrue(legacy_store.import_source(
                "SOUL.md", legacy_hash, [text], [[1.0, 0.0]], "instruction", 0.9,
                sections=["Legacy"], source_sha256=legacy_hash, import_version=1,
            ))
            legacy_store._get_conn().close()

            provider = plugin.OllamaMemoryProvider()
            provider.initialize(session_id="chunker-upgrade", hermes_home=td)
            rows = self.search_all(provider)
            versions = {row["metadata"]["import_version"] for row in rows}
            self.assertEqual(versions, {1, 2})
            new_rows = [row for row in rows if row["metadata"]["import_version"] == 2]
            self.assertGreater(len(new_rows), 1)
            self.assertTrue(all(len(row["content"]) <= plugin._IMPORT_CHUNK_CHARS for row in new_rows))
            provider.shutdown()

    def test_prefetch_falls_back_to_exact_lexical_match_on_embedding_timeout(self):
        TimeoutOllamaClient.timeouts.clear()
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(plugin, "_OllamaClient", TimeoutOllamaClient),
            patch.object(plugin, "_load_plugin_config", return_value={}),
        ):
            provider = plugin.OllamaMemoryProvider()
            provider.initialize(session_id="timeout-fallback", hermes_home=td)
            content = "Old profile text: " + ("background detail " * 40) + "Elisabetta phone detail: 555-0102"
            provider._store.add(content, [0.0, 1.0], memory_type="preference")
            block = provider.prefetch("Elisabetta phone detail 555-0102")
            self.assertIn("exact lexical match", block)
            self.assertIn("Elisabetta phone detail: 555-0102", block)
            self.assertTrue(TimeoutOllamaClient.timeouts)
            self.assertLessEqual(max(TimeoutOllamaClient.timeouts), 6.0)
            provider.shutdown()

    def test_failed_embedding_leaves_no_partial_import_and_retries(self):
        FailOnceOllamaClient.fail_next_embedding = True
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(plugin, "_OllamaClient", FailOnceOllamaClient),
            patch.object(plugin, "_load_plugin_config", return_value={"import_profile_files": True}),
        ):
            home = Path(td)
            (home / "SOUL.md").write_text("x" * 2000, encoding="utf-8")

            first = plugin.OllamaMemoryProvider()
            first.initialize(session_id="failed-embedding", hermes_home=str(home))
            first_status = json.loads(first.handle_tool_call("ollama_status", {}))
            self.assertEqual(first_status["count"], 0)
            self.assertEqual(first_status["imported_sources"], 0)
            first.shutdown()

            retry = plugin.OllamaMemoryProvider()
            retry.initialize(session_id="retry-embedding", hermes_home=str(home))
            retry_status = json.loads(retry.handle_tool_call("ollama_status", {}))
            self.assertEqual(retry_status["count"], len(plugin._split_for_embedding("x" * 2000)))
            self.assertEqual(retry_status["imported_sources"], 1)
            retry.shutdown()

    def test_import_is_disabled_without_explicit_opt_in(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(plugin, "_OllamaClient", FakeOllamaClient),
            patch.object(plugin, "_load_plugin_config", return_value={}),
        ):
            home = Path(td)
            (home / "memories").mkdir()
            (home / "memories" / "MEMORY.md").write_text("Private profile fact.", encoding="utf-8")
            provider = plugin.OllamaMemoryProvider()
            provider.initialize(session_id="opt-out", hermes_home=str(home))
            status = json.loads(provider.handle_tool_call("ollama_status", {}))
            self.assertEqual(status["count"], 0)
            self.assertEqual(status["imported_sources"], 0)
            provider.shutdown()

    def test_rejects_profile_symlink_outside_active_home(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(plugin, "_OllamaClient", FakeOllamaClient),
            patch.object(plugin, "_load_plugin_config", return_value={"import_profile_files": True}),
        ):
            root = Path(td)
            home = root / "profile"
            home.mkdir()
            outside = root / "outside.md"
            outside.write_text("Must not be imported from outside the profile.", encoding="utf-8")
            try:
                (home / "SOUL.md").symlink_to(outside)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            provider = plugin.OllamaMemoryProvider()
            provider.initialize(session_id="symlink", hermes_home=str(home))
            status = json.loads(provider.handle_tool_call("ollama_status", {}))
            self.assertEqual(status["count"], 0)
            self.assertEqual(status["imported_sources"], 0)
            provider.shutdown()


if __name__ == "__main__":
    unittest.main()
