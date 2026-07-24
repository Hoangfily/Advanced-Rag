from pathlib import Path

from pypdf import PdfReader


def load_document(path: str) -> str:
    if Path(path).suffix.lower() == ".pdf":
        return _load_pdf(path)
    return Path(path).read_text(encoding="utf-8")


def _load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(paths: list[str]) -> dict[str, str]:
    return {path: load_document(path) for path in paths}
