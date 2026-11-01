"""Formatação do contexto recuperado (sem dependências pesadas)."""
from typing import Iterable, Tuple


def format_context(results: Iterable[Tuple[object, float]], max_chars: int) -> str:
    """Monta o bloco de contexto com a página de cada trecho, respeitando um limite de tamanho."""
    parts, total = [], 0
    for doc, _score in results:
        page = doc.metadata.get("page", "?")
        block = f"[Página {page}]\n{doc.page_content.strip()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)
