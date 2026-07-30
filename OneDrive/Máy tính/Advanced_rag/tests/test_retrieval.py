from app.retrieval.reranker import Reranker
from app.schemas.models import RetrievedChunk


def test_reranker_ranks_relevant_text_higher():
    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = [
        RetrievedChunk(chunk_id="a", text="Con mèo đang ngủ trên ghế sofa.", source="doc.txt", score=0.0),
        RetrievedChunk(
            chunk_id="b",
            text="Python là ngôn ngữ lập trình phổ biến cho machine learning.",
            source="doc.txt",
            score=0.0,
        ),
        RetrievedChunk(chunk_id="c", text="Con mèo đang ngủ trên ghế sofa.", source="doc.txt", score=0.0),
    ]

    reranked = reranker.rerank("ngôn ngữ lập trình nào dùng cho machine learning?", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "b"
    assert len({c.chunk_id for c in reranked}) == len(reranked)
