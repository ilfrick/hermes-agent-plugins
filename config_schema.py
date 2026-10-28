"""Declarative config schema for Ollama Memory Provider.

This file is loaded by the web dashboard to render the configuration panel.
"""

from __future__ import annotations

from plugins.memory.config_schema import (
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
    KIND_TEXT,
    KIND_NUMBER,
    KIND_BOOL,
    STORAGE_FLAT_JSON,
)


CONFIG_SCHEMA = ProviderConfigSchema(
    name="ollama_memory",
    label="Ollama Local Embeddings",
    storage=STORAGE_FLAT_JSON,
    docs_url="https://github.com/NousResearch/hermes-agent/blob/main/docs/user-guide/features/memory.md#ollama-memory",
    fields=(
        ProviderField(
            key="endpoint",
            label="Ollama API Endpoint",
            kind=KIND_TEXT,
            default="http://localhost:11434",
            description="Base URL of the Ollama server (must include protocol and port).",
            placeholder="http://host:11434",
            inline=True,
            group="Connection",
        ),
        ProviderField(
            key="model",
            label="Embedding Model",
            kind=KIND_TEXT,
            default="qwen3-embedding:4b",
            description="Name of the embedding model as shown in `ollama list`. Must support `/api/embed` (legacy `/api/embeddings` is also supported).",
            placeholder="qwen3-embedding:4b",
            inline=True,
            group="Connection",
        ),
        ProviderField(
            key="collection",
            label="Collection Name",
            kind=KIND_TEXT,
            default="hermes_memory",
            description="Logical collection name (used as SQLite table suffix). Changing this creates a new isolated store.",
            placeholder="hermes_memory",
            inline=True,
            group="Storage",
        ),
        ProviderField(
            key="top_k",
            label="Default Top-K Results",
            kind=KIND_NUMBER,
            default="8",
            description="Default number of search results returned by ollama_search.",
            inline=True,
            group="Search",
        ),
        ProviderField(
            key="similarity_threshold",
            label="Similarity Threshold",
            kind=KIND_NUMBER,
            default="0.25",
            description="Cosine similarity cutoff (0-1). Lower = more results, higher = stricter relevance.",
            placeholder="0.25",
            inline=True,
            group="Search",
        ),
        ProviderField(
            key="max_context_tokens",
            label="Max Context Tokens",
            kind=KIND_NUMBER,
            default="1200",
            description="Maximum tokens in the prefetch context block injected into the system prompt.",
            inline=True,
            group="Search",
        ),
        ProviderField(
            key="keep_alive",
            label="Embedding Model Keep-Alive",
            kind=KIND_TEXT,
            default="10m",
            description="Ask Ollama to keep the embedding model loaded for this long after a request.",
            placeholder="10m",
            inline=True,
            group="Connection",
        ),
        ProviderField(
            key="prefetch_embedding_timeout",
            label="Prefetch Embedding Timeout (seconds)",
            kind=KIND_NUMBER,
            default="5",
            description="Maximum wait for turn-start embeddings; capped at 6 seconds so lexical fallback can run before the gateway timeout.",
            inline=True,
            group="Search",
        ),
        ProviderField(
            key="auto_embed_writes",
            label="Auto-embed Built-in Memory Writes",
            kind=KIND_BOOL,
            default="true",
            description="When true, every memory.add / memory.replace call is also embedded and stored in Ollama memory.",
            inline=True,
            group="Behavior",
        ),
        ProviderField(
            key="import_profile_files",
            label="Import Profile Files (opt-in)",
            kind=KIND_BOOL,
            default="false",
            description="When enabled, the contents of MEMORY.md, USER.md, and SOUL.md are sent to the configured Ollama endpoint and added to this profile's local vector database.",
            inline=True,
            group="Behavior",
        ),
    ),
)
