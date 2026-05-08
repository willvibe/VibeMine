from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
import logging
import time
from app.config import CORS_ORIGINS
from app.routes.upload import router as upload_router
from app.routes.train import router as train_router
from app.routes.download import router as download_router
from app.routes.ai import router as ai_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VibeMine API",
    description="在线智能数据挖掘与 AutoML 平台",
    version="1.0.0",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(train_router)
app.include_router(download_router)
app.include_router(ai_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "VibeMine API"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.1f}ms")
    return response