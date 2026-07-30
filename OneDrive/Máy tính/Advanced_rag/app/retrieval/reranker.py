from sentence_transformers import CrossEncoder

from app.schemas.models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        unique_candidates = {c.chunk_id: c for c in candidates}.values()
        candidates = list(unique_candidates)
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [c.model_copy(update={"score": float(score)}) for c, score in ranked[:top_k]]
