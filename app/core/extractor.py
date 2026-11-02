import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import settings
from app.core.context import format_context
from app.prompt.systemPrompts import (
    EXTRACTION_QUERIES,
    EXTRACTION_SYSTEM_PROMPT,
    REPAIR_PROMPT,
    build_extraction_user_prompt,
)


class ExtractionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Esquema de saída
# --------------------------------------------------------------------------- #
class Etapa(BaseModel):
    etapa: str
    data: Optional[str] = None


_TEXT_FIELDS = (
    "numero_edital", "orgao", "titulo", "objeto", "publico_alvo",
    "vagas", "valor_beneficio", "como_se_inscrever", "recursos_e_contato",
)
_LIST_FIELDS = ("requisitos", "documentos_exigidos", "criterios_selecao")


class Edital(BaseModel):
    numero_edital: Optional[str] = None
    orgao: Optional[str] = None
    titulo: Optional[str] = None
    objeto: Optional[str] = None
    publico_alvo: Optional[str] = None
    vagas: Optional[str] = None
    valor_beneficio: Optional[str] = None
    cronograma: List[Etapa] = Field(default_factory=list)
    requisitos: List[str] = Field(default_factory=list)
    documentos_exigidos: List[str] = Field(default_factory=list)
    criterios_selecao: List[str] = Field(default_factory=list)
    como_se_inscrever: Optional[str] = None
    recursos_e_contato: Optional[str] = None

    # LLMs às vezes devolvem número, lista ou "null" como texto: normalizamos.
    @field_validator(*_TEXT_FIELDS, mode="before")
    @classmethod
    def _norm_text(cls, v):
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            v = "; ".join(str(x) for x in v)
        v = str(v).strip()
        return None if v.lower() in ("", "null", "none", "n/a", "não informado") else v

    @field_validator(*_LIST_FIELDS, mode="before")
    @classmethod
    def _norm_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v if str(x).strip()]

    @field_validator("cronograma", mode="before")
    @classmethod
    def _norm_cronograma(cls, v):
        if v is None:
            return []
        return [{"etapa": x} if isinstance(x, str) else x for x in v]


@dataclass
class ExtractionResult:
    edital: Edital
    pages: List[int]  # páginas que serviram de contexto
    n_chunks: int


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def parse_json_response(text: str) -> dict:
    """Extrai o objeto JSON de uma resposta de LLM (tolera ```json e texto extra)."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("nenhum objeto JSON encontrado")
    return json.loads(text[start : end + 1])


def build_checklist_md(ed: Edital) -> str:
    """Checklist em Markdown gerado a partir dos dados estruturados (sem nova chamada ao LLM)."""
    lines = [f"# Checklist — {ed.titulo or ed.numero_edital or 'Edital'}", ""]
    if ed.cronograma:
        lines += ["## Prazos"] + [f"- [ ] {e.etapa}: {e.data or 'sem data'}" for e in ed.cronograma] + [""]
    if ed.requisitos:
        lines += ["## Requisitos / elegibilidade"] + [f"- [ ] {r}" for r in ed.requisitos] + [""]
    if ed.documentos_exigidos:
        lines += ["## Documentos exigidos"] + [f"- [ ] {d}" for d in ed.documentos_exigidos] + [""]
    return "\n".join(lines)


def extraction_path(doc_id: str) -> Path:
    return Path(settings.EXTRACTION_DIR) / f"{doc_id}.json"


def load_saved(doc_id: str) -> Optional[Edital]:
    p = extraction_path(doc_id)
    return Edital.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None


def save_extraction(doc_id: str, ed: Edital) -> None:
    extraction_path(doc_id).write_text(ed.model_dump_json(indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
class EditalExtractor:
    def __init__(self, rag, llm):
        self.rag, self.llm = rag, llm

    def extract(self, doc_id: str) -> ExtractionResult:
        results = self.rag.retrieve_many(doc_id, EXTRACTION_QUERIES, k_each=settings.EXTRACT_K)
        if not results:
            raise ExtractionError("Nenhum trecho encontrado. O PDF foi indexado?")
        context = format_context(results, settings.MAX_CONTEXT_CHARS)

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_extraction_user_prompt(context)},
        ]
        raw = self.llm.complete(messages, temperature=0.0, max_tokens=4096)

        edital = None
        for attempt in range(2):  # 1 tentativa + 1 reparo
            try:
                edital = Edital.model_validate(parse_json_response(raw))
                break
            except (ValueError, ValidationError) as e:
                if attempt == 1:
                    raise ExtractionError(f"O modelo não devolveu um JSON válido: {e}") from e
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": REPAIR_PROMPT.format(erro=str(e)[:200])},
                ]
                raw = self.llm.complete(messages, temperature=0.0, max_tokens=4096)

        pages = sorted({d.metadata.get("page") for d, _ in results if d.metadata.get("page")})
        return ExtractionResult(edital, pages, len(results))
