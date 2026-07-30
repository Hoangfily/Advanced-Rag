from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    score: float


class QueryResponse(BaseModel):
    answer: str
    sub_queries: list[str] = []
    contexts: list[RetrievedChunk] = []


class IngestRequest(BaseModel):
    paths: list[str]
