import importlib.util
import json
import sys
import tempfile
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
setattr(memory_api, "spawn_context_thread", lambda fn, *a, **kw: fn(*a, **kw))
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
    def __init__(self, endpoint, model):
        self.endpoint = endpoint
        self.model = model

    def embed(self, text):
        return [1.0, 0.0]

    def health_check(self):
        return True


class FailOnceOllamaClient(FakeOllamaClient):
    fail_next_embedding = True

    def embed(self, text):
        if type(self).fail_next_embedding:
            type(self).fail_next_embedding = False
            raise RuntimeError("simulated embedding failure")
        return super().embed(text)


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
            self.assertIn(source_text["memories/USER.md"], contents)
            self.assertIn(changed_text, contents)
            third.shutdown()


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
            self.assertEqual(retry_status["count"], 2)
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
