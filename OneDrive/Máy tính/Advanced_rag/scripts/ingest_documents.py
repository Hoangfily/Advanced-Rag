import argparse

from app.pipeline.rag_pipeline import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Đường dẫn các file cần ingest")
    args = parser.parse_args()

    pipeline = RagPipeline()
    count = pipeline.ingest(args.paths)
    print(f"Đã ingest {count} chunks.")


if __name__ == "__main__":
    main()
