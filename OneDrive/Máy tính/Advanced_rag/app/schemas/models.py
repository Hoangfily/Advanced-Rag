from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sub_queries: list[str] = []
    contexts: list[RetrievedChunk] = []


class IngestRequest(BaseModel):
    paths: list[str]
