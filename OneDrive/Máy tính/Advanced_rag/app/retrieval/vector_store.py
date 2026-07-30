from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from app.ingestion.chunking import Chunk
from app.ingestion.embedder import Embedder
from app.schemas.models import RetrievedChunk


class _EmbedderAdapter(Embeddings):
    """Bridges the Gemini-backed Embedder into the langchain Embeddings interface."""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str, embedder: Embedder):
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=_EmbedderAdapter(embedder),
            persist_directory=persist_dir,
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        documents = [_chunk_to_document(c) for c in chunks]
        self._store.add_documents(documents, ids=[c.chunk_id for c in chunks])

    def query(self, query: str, top_k: int) -> list[RetrievedChunk]:
        results = self._store.similarity_search_with_score(query, k=top_k)
        return [_document_to_chunk(doc, score) for doc, score in results]

    def as_retriever(self, top_k: int) -> BaseRetriever:
        return self._store.as_retriever(search_kwargs={"k": top_k})

    def get_all_chunks(self) -> list[Chunk]:
        result = self._store.get()
        return [
            Chunk(chunk_id=result["ids"][i], text=result["documents"][i], source=result["metadatas"][i]["source"])
            for i in range(len(result["ids"]))
        ]


def _chunk_to_document(chunk: Chunk) -> Document:
    return Document(page_content=chunk.text, metadata={"source": chunk.source, "chunk_id": chunk.chunk_id})


def _document_to_chunk(document: Document, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=document.metadata["chunk_id"],
        text=document.page_content,
        source=document.metadata["source"],
        score=float(score),
    )
