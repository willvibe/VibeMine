import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.data_service import save_upload_file, parse_csv, get_data_profile
from app.services.ai_service import get_data_insight
from app.config import UPLOAD_DIR

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="仅支持 CSV 文件上传")

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="上传文件为空")

        saved_name = save_upload_file(file_bytes, file.filename)
        df = parse_csv(saved_name)
        profile = get_data_profile(df)

        ai_insight = get_data_insight(profile)

        return {
            "filename": saved_name,
            "columns": profile["column_names"],
            "shape": profile["shape"],
            "preview": profile["preview"],
            "column_details": profile["columns"],
            "ai_insight": ai_insight,
            "imbalance_detected": profile.get("imbalance_detected", False),
            "imbalance_ratio": profile.get("imbalance_ratio", 0.0),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件解析失败: {str(e)}")