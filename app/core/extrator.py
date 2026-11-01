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


@dataclass
class ExtractionResult:
    edital: Edital
    pages: List[int]  
    n_chunks: int