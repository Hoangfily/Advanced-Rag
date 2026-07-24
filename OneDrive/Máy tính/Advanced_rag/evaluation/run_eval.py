import json
from pathlib import Path

from app.pipeline.rag_pipeline import RagPipeline
from evaluation.metrics import mean_reciprocal_rank, recall_at_k


def main() -> None:
    eval_file = Path("evaluation/eval_dataset/qa_pairs.json")
    qa_pairs = json.loads(eval_file.read_text(encoding="utf-8"))

    pipeline = RagPipeline()
    recalls, mrrs = [], []

    for item in qa_pairs:
        result = pipeline.run(item["question"])
        retrieved_ids = [c.chunk_id for c in result.contexts]
        recalls.append(recall_at_k(retrieved_ids, item["relevant_chunk_ids"], k=5))
        mrrs.append(mean_reciprocal_rank(retrieved_ids, item["relevant_chunk_ids"]))

    print(f"Recall@5: {sum(recalls) / len(recalls):.4f}")
    print(f"MRR: {sum(mrrs) / len(mrrs):.4f}")


if __name__ == "__main__":
    main()
