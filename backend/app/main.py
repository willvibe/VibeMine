from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.config import CORS_ORIGINS
from app.routes.upload import router as upload_router
from app.routes.train import router as train_router
from app.routes.download import router as download_router

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


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "VibeMine API"}