from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str


def chunk_text(text: str, source: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(Chunk(chunk_id=f"{source}::{index}", text=text[start:end], source=source))
        start += chunk_size - chunk_overlap
        index += 1
    return chunks
