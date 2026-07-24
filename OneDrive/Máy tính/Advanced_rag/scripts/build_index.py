from pathlib import Path

from app.pipeline.rag_pipeline import RagPipeline


def main() -> None:
    raw_dir = Path("data/raw")
    paths = [str(p) for p in raw_dir.glob("**/*") if p.is_file()]

    pipeline = RagPipeline()
    count = pipeline.ingest(paths)
    print(f"Đã build index với {count} chunks từ {len(paths)} file.")


if __name__ == "__main__":
    main()
