from rank_bm25 import BM25Okapi

from app.ingestion.chunking import Chunk


class SparseRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([c.text.split() for c in chunks])

    def query(self, text: str, top_k: int) -> list[dict]:
        scores = self.bm25.get_scores(text.split())
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [
            {"chunk_id": chunk.chunk_id, "text": chunk.text, "source": chunk.source, "score": score}
            for chunk, score in ranked
        ]
