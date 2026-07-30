from app.pipeline.rag_pipeline import RagPipeline


def main() -> None:
    pipeline = RagPipeline()

    print("1. Ingest data/raw/FSoft_HR.pdf...")
    count = pipeline.ingest(["data/raw/FSoft_HR.pdf"])
    print(f"   OK - {count} chunks.")

    question = "Chính sách nhân sự của công ty có những nội dung chính nào?"
    print(f"2. Query: {question}")
    result = pipeline.run(question)

    print(f"   Sub-queries: {result.sub_queries}")
    print(f"   Answer: {result.answer}")
    print("   Contexts:")
    for chunk in result.contexts:
        print(f"     - [{chunk.source}] score={chunk.score:.4f} | {chunk.text[:80]}...")


if __name__ == "__main__":
    main()
