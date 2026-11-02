
import re
from typing import Dict, Iterator, List

from openai import OpenAI

from app.core.config import settings

Message = Dict[str, str]

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


class LLMError(RuntimeError):
    """Erro amigável para mostrar ao usuário na interface."""


class LLMClient:
    def __init__(self):
        provider = settings.LLM_PROVIDER
        self.provider = provider
        self.extra_body = None

        if provider == "nvidia":
            if not settings.NVIDIA_API_KEY:
                raise LLMError("NVIDIA_API_KEY não definida. Copie .env.example para .env e preencha a chave.")
            base_url, api_key, self.model = settings.NVIDIA_BASE_URL, settings.NVIDIA_API_KEY, settings.NVIDIA_MODEL
            self.extra_body = {"chat_template_kwargs": {"enable_thinking": settings.LLM_ENABLE_THINKING}}
        elif provider == "huggingface":
            if not settings.HF_TOKEN:
                raise LLMError("HF_TOKEN não definido no .env.")
            base_url, api_key, self.model = settings.HF_BASE_URL, settings.HF_TOKEN, settings.HF_MODEL
        elif provider == "ollama":
            base_url = settings.OLLAMA_BASE_URL.rstrip("/") + "/v1"
            api_key, self.model = "ollama", settings.OLLAMA_MODEL
        else:
            raise LLMError(f"LLM_PROVIDER inválido: '{provider}'. Use nvidia, huggingface ou ollama.")

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=settings.LLM_TIMEOUT)

    # ------------------------------------------------------------------ #
    def _kwargs(self, messages: List[Message], temperature: float, max_tokens: int, stream: bool) -> dict:
        kw = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        return kw

    @staticmethod
    def _clean(text: str) -> str:
        return _THINK_RE.sub("", text or "").strip()

    # ------------------------------------------------------------------ #
    def complete(self, messages: List[Message], temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """Resposta completa (usada na extração estruturada)."""
        try:
            resp = self.client.chat.completions.create(**self._kwargs(messages, temperature, max_tokens, False))
        except Exception as e:  # rede, chave inválida, rate limit...
            raise LLMError(f"Falha ao chamar o LLM ({self.provider}/{self.model}): {e}") from e
        return self._clean(resp.choices[0].message.content)

    def stream(self, messages: List[Message], temperature: float = 0.1, max_tokens: int = 2048) -> Iterator[str]:
        """Streaming token a token (usado no chat). Ignora o 'reasoning_content'."""
        try:
            stream = self.client.chat.completions.create(**self._kwargs(messages, temperature, max_tokens, True))
            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    yield piece
        except Exception as e:
            raise LLMError(f"Falha ao chamar o LLM ({self.provider}/{self.model}): {e}") from e

    def describe(self) -> str:
        return f"{self.provider} · {self.model}"
