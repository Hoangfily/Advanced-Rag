from fastapi import APIRouter, Depends

from app.api.deps import get_pipeline
from app.pipeline.rag_pipeline import RagPipeline
from app.schemas.models import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def run_query(request: QueryRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> QueryResponse:
    return pipeline.run(request.question, top_k=request.top_k)
