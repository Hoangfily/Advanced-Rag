"""Đánh giá RAG pipeline bằng RAGAS: Faithfulness, Answer Relevancy, Context Precision, Context Recall.

Cách chạy:
    .\\.venv\\Scripts\\python.exe -m evaluation.ragas_eval

Ghi chú kỹ thuật quan trọng:
- `ragas` (mọi phiên bản, kể cả mới nhất) import cứng `langchain_community.chat_models.vertexai
  .ChatVertexAI` ngay khi import. Module này đã bị gỡ khỏi langchain-community trong đợt tái cấu
  trúc "sunset" cho LangChain 1.0 (dự án này chưa từng dùng VertexAI nên cũng không cài đặt nó).
  Ta "giả lập" (stub) module đó trước khi import ragas để việc import không bị lỗi; ragas không
  bao giờ thực sự khởi tạo class này trừ khi mình chủ động yêu cầu dùng VertexAI.
- API "collections" mới của ragas (Faithfulness/AnswerRelevancy/ContextPrecision/ContextRecall)
  cần một LLM kiểu InstructorBaseRagasLLM, KHÔNG tương thích với LangchainLLMWrapper cũ. Ta dùng
  `llm_factory` của ragas trỏ tới endpoint tương thích OpenAI mà Google cung cấp sẵn cho Gemini
  (https://generativelanguage.googleapis.com/v1beta/openai/), tái sử dụng đúng GEMINI_API_KEY và
  đúng model đã cấu hình trong .env — không cần thêm API key hay provider nào khác.
- Gemini free tier giới hạn khá thấp request/phút, nên script tự retry (có chờ) khi gặp lỗi 429.
"""
import asyncio
import csv
import json
import sys
import time
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - never instantiated, import-time shim only
        pass

    _vertexai_stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from google import genai
from openai import AsyncOpenAI
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from app.core.config import get_settings
from app.pipeline.rag_pipeline import RagPipeline

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 25


async def with_retry(coro_fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  (lỗi tạm thời, thử lại sau {RETRY_WAIT_SECONDS}s: {type(e).__name__})")
            await asyncio.sleep(RETRY_WAIT_SECONDS)


def run_pipeline_with_retry(pipeline: RagPipeline, question: str):
    for attempt in range(MAX_RETRIES):
        try:
            return pipeline.run(question)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  (lỗi tạm thời khi chạy pipeline, thử lại sau {RETRY_WAIT_SECONDS}s: {type(e).__name__})")
            time.sleep(RETRY_WAIT_SECONDS)


async def main() -> None:
    settings = get_settings()
    qa_pairs = json.loads(open("evaluation/eval_dataset/qa_pairs.json", encoding="utf-8").read())

    pipeline = RagPipeline()

    openai_client = AsyncOpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_BASE_URL)
    ragas_llm = llm_factory(settings.llm_model, client=openai_client)
    ragas_embeddings = GoogleEmbeddings(client=genai.Client(api_key=settings.gemini_api_key), model=settings.embedding_model)

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    context_precision = ContextPrecision(llm=ragas_llm)
    context_recall = ContextRecall(llm=ragas_llm)

    rows = []
    for i, item in enumerate(qa_pairs, start=1):
        print(f"[{i}/{len(qa_pairs)}] {item['question']}")

        result = run_pipeline_with_retry(pipeline, item["question"])
        contexts = [c.text for c in result.contexts]
        reference = item["ground_truth"]

        f_score = await with_retry(
            faithfulness.ascore, user_input=item["question"], response=result.answer, retrieved_contexts=contexts
        )
        ar_score = await with_retry(answer_relevancy.ascore, user_input=item["question"], response=result.answer)
        cp_score = await with_retry(
            context_precision.ascore, user_input=item["question"], reference=reference, retrieved_contexts=contexts
        )
        cr_score = await with_retry(
            context_recall.ascore, user_input=item["question"], retrieved_contexts=contexts, reference=reference
        )

        rows.append(
            {
                "question": item["question"],
                "answer": result.answer,
                "faithfulness": f_score.value,
                "answer_relevancy": ar_score.value,
                "context_precision": cp_score.value,
                "context_recall": cr_score.value,
            }
        )
        print(
            f"  faithfulness={f_score.value:.3f}  answer_relevancy={ar_score.value:.3f}  "
            f"context_precision={cp_score.value:.3f}  context_recall={cr_score.value:.3f}"
        )

    with open("evaluation/ragas_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=== Điểm trung bình ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        avg = sum(r[metric] for r in rows) / len(rows)
        print(f"{metric}: {avg:.4f}")
    print("Chi tiết từng câu hỏi đã lưu tại evaluation/ragas_results.csv")


if __name__ == "__main__":
    asyncio.run(main())
