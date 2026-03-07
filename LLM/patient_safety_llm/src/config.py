import os


class Settings:
    """Configuration for local LLM integration. Set environment variables to override."""
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


settings = Settings()
