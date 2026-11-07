import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "sim")


class Settings:
    PROJECT_NAME: str = "SMART EDITAL"

    # ---------- Armazenamento ----------
    UPLOAD_DIR: str = str(BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads"))
    CHROMA_DIR: str = str(BASE_DIR / os.getenv("CHROMA_DIR", "data/chroma"))
    EXTRACTION_DIR: str = str(BASE_DIR / os.getenv("EXTRACTION_DIR", "data/extractions"))

    # ---------- LLM ----------NVIDIA 
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "nvidia").lower()

    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    NVIDIA_MODEL: str = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

    LLM_ENABLE_THINKING: bool = _bool("LLM_ENABLE_THINKING", False)

    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    HF_BASE_URL: str = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
    HF_MODEL: str = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "120"))

    # ---------- RAG ----------
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    EXTRACT_K: int = int(os.getenv("EXTRACT_K", "3"))  # chunks por pergunta na extração
    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))
    HISTORY_TURNS: int = int(os.getenv("HISTORY_TURNS", "4"))


settings = Settings()

for _d in (settings.UPLOAD_DIR, settings.CHROMA_DIR, settings.EXTRACTION_DIR):
    os.makedirs(_d, exist_ok=True)
