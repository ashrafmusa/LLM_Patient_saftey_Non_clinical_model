import os
from pathlib import Path


class Settings:
    """Configuration for local LLM integration. Set environment variables to override."""
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Example: http://127.0.0.1:9000  (llama.cpp server) - no trailing slash
    LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL")
    # Example: path to local llama binary (llama.cpp front-end). Default tries `llama` on PATH.
    LLAMA_BINARY = os.getenv("LLAMA_BINARY", "llama")
    # Timeout seconds for external calls
    LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", "30"))

    # OpenAI-compatible API settings (works with OpenAI, Azure OpenAI-compatible,
    # and local gateways that expose /v1/chat/completions).
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))

    # Retrieval grounding
    RETRIEVAL_ENABLED = os.getenv("RETRIEVAL_ENABLED", "true").lower() in {"1", "true", "yes"}
    RETRIEVAL_DOCS_DIR = os.getenv("RETRIEVAL_DOCS_DIR", str(_PROJECT_ROOT / "docs"))
    RETRIEVAL_MAX_CHUNKS = int(os.getenv("RETRIEVAL_MAX_CHUNKS", "3"))

    # Audit logging
    AUDIT_LOG_DIR = os.getenv("AUDIT_LOG_DIR", str(_PROJECT_ROOT / "reports" / "audit_logs"))


settings = Settings()
