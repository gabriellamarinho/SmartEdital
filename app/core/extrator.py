import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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