import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


@dataclass
class ProcessedPDF:
    doc_id: str
    filename: str
    chunks: List[Document]
    n_pages: int
    n_chars: int
    is_scanned: bool  # PDF sem camada de texto (precisaria de OCR)


def compute_doc_id(data: bytes) -> str:
    """ID estável do documento (hash do conteúdo): mesmo PDF = mesma coleção no Chroma."""
    return hashlib.sha256(data).hexdigest()[:16]


def save_upload(data: bytes, filename: str) -> Tuple[Path, str]:
    doc_id = compute_doc_id(data)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    path = Path(settings.UPLOAD_DIR) / f"{doc_id}_{safe}"
    if not path.exists():
        path.write_bytes(data)
    return path, doc_id


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-zà-ú])", "", text)  # palavras hifenizadas na quebra de linha
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_pdf(file_path: str | Path, doc_id: str, filename: str) -> ProcessedPDF:
    """Lê o PDF, limpa o texto, preserva página como metadado e faz o chunking."""
    pages = PyPDFLoader(str(file_path)).load()
    for p in pages:
        p.page_content = _clean(p.page_content)
        p.metadata = {"source": filename, "doc_id": doc_id, "page": int(p.metadata.get("page", 0)) + 1}

    n_chars = sum(len(p.page_content) for p in pages)
    is_scanned = n_chars < 50 * max(len(pages), 1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "Art. ", ". ", " ", ""],
    )
    chunks = splitter.split_documents([p for p in pages if p.page_content])
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    return ProcessedPDF(doc_id, filename, chunks, len(pages), n_chars, is_scanned)
