from langchain_classic.retrievers.ensemble import EnsembleRetriever
from pydantic import BaseModel, Field

from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.vector_store import VectorStore
from app.schemas.models import RetrievedChunk


class HybridRetrieverConfig(BaseModel):
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k_dense: int = Field(default=20, gt=0)


class HybridRetriever:
    """Fuses dense (Chroma) and sparse (BM25) retrieval via weighted reciprocal rank fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        sparse_retriever: SparseRetriever,
        alpha: float = 0.5,
        top_k_dense: int = 20,
    ):
        config = HybridRetrieverConfig(alpha=alpha, top_k_dense=top_k_dense)
        self._ensemble = EnsembleRetriever(
            retrievers=[vector_store.as_retriever(config.top_k_dense), sparse_retriever.as_retriever()],
            weights=[config.alpha, 1.0 - config.alpha],
            id_key="chunk_id",
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        documents = self._ensemble.invoke(query)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=doc.metadata["chunk_id"],
                text=doc.page_content,
                source=doc.metadata["source"],
                score=1.0 / (rank + 1),
            )
            for rank, doc in enumerate(documents)
        ]
