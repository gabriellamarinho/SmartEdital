from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import settings

def process_pdf(file_path: str) -> List[Document]:
    """Lê o PDF, preserva metadados (como página) e faz o chunking do texto."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "Art.", "Artigo", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks