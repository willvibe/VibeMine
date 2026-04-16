import os
import re
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from app.services.data_service import save_upload_file, parse_csv, get_data_profile
from app.services.ai_service import get_data_insight
from app.config import UPLOAD_DIR

router = APIRouter(prefix="/api", tags=["upload"])
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_ROWS = 50000
MAX_COLUMNS = 200


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="仅支持 CSV 文件上传，请上传 .csv 格式文件")

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="上传文件为空，请选择包含数据的 CSV 文件")

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大（{len(file_bytes) / 1024 / 1024:.1f}MB），请控制在 100MB 以内",
            )

        saved_name = save_upload_file(file_bytes, file.filename)
        try:
            df = parse_csv(saved_name)
        except Exception as e:
            _cleanup_file(saved_name)
            raise HTTPException(
                status_code=400,
                detail="CSV 文件解析失败，请检查文件格式是否正确",
            )

        rows, cols = df.shape
        if rows > MAX_ROWS:
            _cleanup_file(saved_name)
            raise HTTPException(
                status_code=413,
                detail=f"数据量过大（{rows} 行），请控制在 {MAX_ROWS} 行以内以保证训练稳定性",
            )
        if cols > MAX_COLUMNS:
            _cleanup_file(saved_name)
            raise HTTPException(
                status_code=413,
                detail=f"特征列过多（{cols} 列），请控制在 {MAX_COLUMNS} 列以内",
            )
        if rows < 3:
            _cleanup_file(saved_name)
            raise HTTPException(
                status_code=400,
                detail=f"数据行数过少（仅 {rows} 行），至少需要 3 行数据才能进行训练",
            )

        profile = get_data_profile(df)

        return {
            "filename": saved_name,
            "columns": profile["column_names"],
            "shape": profile["shape"],
            "preview": profile["preview"],
            "column_details": profile["columns"],
            "info": profile["info"],
            "describe": profile["describe"],
            "total_rows": profile["total_rows"],
            "imbalance_detected": profile.get("imbalance_detected", False),
            "imbalance_ratio": profile.get("imbalance_ratio", 0.0),
            "class_distributions": profile.get("class_distributions", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="文件处理失败，请检查文件格式后重试")


@router.get("/upload/ai-insight/{filename}")
async def get_upload_ai_insight(filename: str, x_api_key: str = Header(None)):
    if not _is_safe_filename(filename):
        raise HTTPException(status_code=400, detail="无效的文件名")

    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    upload_dir_real = os.path.realpath(UPLOAD_DIR)
    if not filepath.startswith(upload_dir_real):
        raise HTTPException(status_code=403, detail="访问被拒绝")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在，请重新上传")
    try:
        df = parse_csv(filename)
        profile = get_data_profile(df)
        insight = get_data_insight(profile, api_key=x_api_key)
        return {"ai_insight": insight}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI insight error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI 分析失败，请稍后重试")


def _is_safe_filename(filename: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9_.-]+$', filename))


def _cleanup_file(saved_name: str):
    try:
        filepath = os.path.join(UPLOAD_DIR, saved_name)
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass
