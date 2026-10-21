from typing import List, Dict
from pypdf import PdfReader
from core.config import settings


def extract_pages(file) -> List[Dict]:
    """Lê o PDF e devolve o texto de cada página."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def split_text(text: str, size: int, overlap: int) -> List[str]:
    """Divide o texto em janelas de `size` caracteres com sobreposição."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def process_pdf(file, source: str) -> List[Dict]:
    """Extrai e divide em chunks, mantendo página e origem."""
    chunks = []
    for page in extract_pages(file):
        for piece in split_text(page["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP):
            chunks.append({"text": piece, "page": page["page"], "source": source})
    return chunks