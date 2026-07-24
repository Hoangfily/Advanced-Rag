from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, sparse_retriever: SparseRetriever, alpha: float = 0.5):
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha

    def retrieve(self, query: str, query_embedding: list[float], top_k: int) -> list[dict]:
        dense_results = self.vector_store.query(query_embedding, top_k)
        sparse_results = self.sparse_retriever.query(query, top_k)
        return _reciprocal_rank_fusion(dense_results, sparse_results)


def _reciprocal_rank_fusion(dense: list[dict], sparse: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}
    for rank_list in (dense, sparse):
        for rank, item in enumerate(rank_list):
            scores[item["chunk_id"]] = scores.get(item["chunk_id"], 0.0) + 1.0 / (k + rank + 1)
            chunks_by_id[item["chunk_id"]] = item
    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunks_by_id[cid] for cid in ranked_ids]
