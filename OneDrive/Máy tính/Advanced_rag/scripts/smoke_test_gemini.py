from app.core.config import get_settings
from app.generation.llm_client import LLMClient
from app.ingestion.embedder import Embedder


def main() -> None:
    settings = get_settings()

    print("1. Test embedding...")
    embedder = Embedder(settings.embedding_model, settings.gemini_api_key)
    vector = embedder.embed_query("Xin chào, đây là câu test embedding.")
    print(f"   OK - vector dài {len(vector)} chiều, 5 giá trị đầu: {vector[:5]}")

    print("2. Test sinh văn bản...")
    llm = LLMClient(settings.llm_model, settings.gemini_api_key)
    answer = llm.complete("Trả lời ngắn gọn bằng 1 câu: RAG là gì?")
    print(f"   OK - phản hồi: {answer}")


if __name__ == "__main__":
    main()
