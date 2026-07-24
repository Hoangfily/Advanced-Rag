import chromadb

from app.ingestion.chunking import Chunk


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source} for c in chunks],
        )

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return _format_result(result)

    def get_all_chunks(self) -> list[Chunk]:
        result = self.collection.get()
        return [
            Chunk(chunk_id=result["ids"][i], text=result["documents"][i], source=result["metadatas"][i]["source"])
            for i in range(len(result["ids"]))
        ]


def _format_result(result: dict) -> list[dict]:
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"chunk_id": ids[i], "text": documents[i], "source": metadatas[i]["source"], "score": distances[i]}
        for i in range(len(ids))
    ]
