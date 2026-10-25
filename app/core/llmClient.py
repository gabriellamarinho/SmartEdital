from langchain_community.llms import Ollama
from app.core.config import settings
from app.prompts.system_prompts import SYSTEM_RAG_PROMPT, CHECKLIST_PROMPT

class LLMClient:
    def __init__(self):
        if settings.LLM_PROVIDER == "ollama":
            self.llm = Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=0.1
            )
        else:
            # Caso queira expandir para NVIDIA NIM ou Hugging Face API futuramente
            raise NotImplementedError(f"Provedor {settings.LLM_PROVIDER} ainda não configurado.")

    def generate_response(self, question: str, context_docs: list) -> str:
        """Gera resposta via RAG utilizando o contexto extraído."""
        context_text = "\n\n".join(
            [f"[Página {doc.metadata.get('page', 0) + 1}]: {doc.page_content}" for doc in context_docs]
        )
        
        prompt = SYSTEM_RAG_PROMPT.format(context=context_text, question=question)
        return self.llm.invoke(prompt)

    def generate_checklist(self, context_docs: list) -> str:
        """Gera um checklist automático com base no contexto completo extraído."""
        context_text = "\n\n".join([doc.page_content for doc in context_docs[:10]])  # Primeiros trechos chave
        prompt = CHECKLIST_PROMPT.format(context=context_text)
        return self.llm.invoke(prompt)