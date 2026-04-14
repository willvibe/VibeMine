import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.config import MODEL_DIR

router = APIRouter(prefix="/api", tags=["download"])


@router.get("/download/{model_id}")
async def download_model(model_id: str):
    pkl_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    if not os.path.exists(pkl_path):
        raise HTTPException(status_code=404, detail="模型文件不存在或已过期")

    return FileResponse(
        path=pkl_path,
        filename=f"vibemine_model_{model_id}.pkl",
        media_type="application/octet-stream",
    )
