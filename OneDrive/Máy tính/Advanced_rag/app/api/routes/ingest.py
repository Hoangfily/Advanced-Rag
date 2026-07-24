from fastapi import APIRouter, Depends

from app.api.deps import get_pipeline
from app.pipeline.rag_pipeline import RagPipeline
from app.schemas.models import IngestRequest

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("")
def ingest_documents(request: IngestRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> dict:
    count = pipeline.ingest(request.paths)
    return {"ingested_chunks": count}
