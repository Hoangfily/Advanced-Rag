from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.ingestion.chunking import Chunk
from app.schemas.models import RetrievedChunk


class SparseRetriever:
    def __init__(self, chunks: list[Chunk], top_k: int = 20):
        documents = [
            Document(page_content=c.text, metadata={"source": c.source, "chunk_id": c.chunk_id}) for c in chunks
        ]
        self._retriever = BM25Retriever.from_documents(documents, preprocess_func=str.split)
        self._retriever.k = top_k

    def as_retriever(self) -> BaseRetriever:
        return self._retriever

    def query(self, text: str, top_k: int) -> list[RetrievedChunk]:
        self._retriever.k = top_k
        documents = self._retriever.invoke(text)
        return [
            RetrievedChunk(
                chunk_id=doc.metadata["chunk_id"],
                text=doc.page_content,
                source=doc.metadata["source"],
                score=1.0 / (rank + 1),
            )
            for rank, doc in enumerate(documents)
        ]
