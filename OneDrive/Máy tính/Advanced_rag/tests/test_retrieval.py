from app.retrieval.hybrid_retriever import _reciprocal_rank_fusion
from app.retrieval.reranker import Reranker


def test_reciprocal_rank_fusion_merges_and_ranks():
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    sparse = [{"chunk_id": "b"}, {"chunk_id": "c"}]

    fused = _reciprocal_rank_fusion(dense, sparse)

    assert fused[0]["chunk_id"] == "b"
    assert {item["chunk_id"] for item in fused} == {"a", "b", "c"}


def test_reranker_ranks_relevant_text_higher():
    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = [
        {"chunk_id": "a", "text": "Con mèo đang ngủ trên ghế sofa.", "source": "doc.txt"},
        {"chunk_id": "b", "text": "Python là ngôn ngữ lập trình phổ biến cho machine learning.", "source": "doc.txt"},
        {"chunk_id": "c", "text": "Con mèo đang ngủ trên ghế sofa.", "source": "doc.txt"},
    ]

    reranked = reranker.rerank("ngôn ngữ lập trình nào dùng cho machine learning?", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "b"
    assert len({c["chunk_id"] for c in reranked}) == len(reranked)
