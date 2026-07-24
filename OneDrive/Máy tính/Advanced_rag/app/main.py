from fastapi import FastAPI

from app.api.routes import health, ingest, query
from app.core.config import get_settings
from app.core.logging import setup_logging

setup_logging()
settings = get_settings()

app = FastAPI(title=settings.app_name)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
