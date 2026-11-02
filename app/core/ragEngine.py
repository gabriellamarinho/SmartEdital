
from typing import Dict, List, Sequence, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings

Result = Tuple[Document, float]  # (trecho, similaridade de cosseno: 1 = idêntico)


class PrefixedEmbeddings(Embeddings):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._inner = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},  
        )
        self._e5 = "e5" in model_name.lower()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(f"query: {text}" if self._e5 else text)


class RAGEngine:
    def __init__(self):
        self.embeddings = PrefixedEmbeddings(settings.EMBEDDING_MODEL)

    # ------------------------------------------------------------------ #
    def _store(self, doc_id: str) -> Chroma:
        return Chroma(
            collection_name=f"edital_{doc_id}",
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def is_indexed(self, doc_id: str) -> bool:
        return self._store(doc_id)._collection.count() > 0

    def index_document(self, doc_id: str, chunks: List[Document]) -> bool:
        """Gera e salva os embeddings. Retorna True se indexou agora, False se já existia."""
        store = self._store(doc_id)
        if store._collection.count() > 0:
            return False
        store.add_documents(chunks, ids=[f"{doc_id}-{c.metadata['chunk_id']}" for c in chunks])
        return True

    # ------------------------------------------------------------------ #
    def query(self, doc_id: str, question: str, k: int = settings.TOP_K) -> List[Result]:
        """Embedding da pergunta -> busca vetorial -> top-k trechos."""
        raw = self._store(doc_id).similarity_search_with_score(question, k=k)
        return [(doc, 1.0 - dist) for doc, dist in raw]  # cosine distance -> similaridade

    def retrieve_many(self, doc_id: str, questions: Sequence[str], k_each: int) -> List[Result]:
        """Várias consultas, sem duplicados, na ordem do documento (usado na extração)."""
        best: Dict[int, Result] = {}
        for q in questions:
            for doc, score in self.query(doc_id, q, k=k_each):
                cid = doc.metadata["chunk_id"]
                if cid not in best or score > best[cid][1]:
                    best[cid] = (doc, score)
        return sorted(best.values(), key=lambda r: r[0].metadata["chunk_id"])

    # ------------------------------------------------------------------ #
    def embed_text(self, text: str) -> List[float]:
        """Embedding de um texto livre (laboratório da interface)."""
        return self.embeddings.embed_query(text)
