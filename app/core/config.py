import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "EditalAI"
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    
    # Provedor do LLM: "ollama", "nvidia", ou "huggingface"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    
    # Configurações de API e Modelos
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Configurações do RAG e Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

settings = Settings()

# Garante que o diretório de uploads exista
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)