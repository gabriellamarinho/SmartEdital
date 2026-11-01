from typing import List, Tuple
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.core.config import settings

class RAGEngine:
    def __init__(self):
        # Modelo de Embeddings local leve e rápido
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        self.vector_store = None

    def create_vector_store(self, chunks: List[Document]) -> Chroma:
        """Cria e armazena os embeddings no ChromaDB temporário na memória."""
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        return self.vector_store

    def query_context(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Realiza busca vetorial no ChromaDB e retorna os k trechos mais parecidos."""
        if not self.vector_store:
            raise ValueError("Nenhum edital foi processado ainda.")
        
        # Retorna documentos acompanhados da pontuação de distância/similaridade
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results