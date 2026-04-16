import os
import re
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.config import MODEL_DIR

router = APIRouter(prefix="/api", tags=["download"])


@router.get("/download/{model_id}")
async def download_model(model_id: str):
    if not re.match(r'^[a-zA-Z0-9_-]+$', model_id):
        raise HTTPException(status_code=400, detail="无效的模型ID")

    pkl_path = os.path.realpath(os.path.join(MODEL_DIR, f"{model_id}.pkl"))
    model_dir_real = os.path.realpath(MODEL_DIR)
    if not pkl_path.startswith(model_dir_real):
        raise HTTPException(status_code=403, detail="访问被拒绝")

    if not os.path.exists(pkl_path):
        raise HTTPException(status_code=404, detail="模型文件不存在或已过期")

    return FileResponse(
        path=pkl_path,
        filename=f"vibemine_model_{model_id}.pkl",
        media_type="application/octet-stream",
    )
