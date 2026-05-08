import os
import threading
import uuid
import logging
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.config import UPLOAD_DIR, MODEL_DIR
from app.services.session_manager import init_session, update_session, get_session, delete_session

router = APIRouter(prefix="/api", tags=["train"])
logger = logging.getLogger(__name__)

MAX_SESSIONS = 50
SESSION_TTL = 3600

class TrainRequest(BaseModel):
    filename: str
    task_type: str
    target_column: str = ""
    selected_models: Optional[List[str]] = None
    ignore_columns: Optional[List[str]] = None
    use_smote: bool = False
    use_outlier_removal: bool = True
    use_advanced_imputation: bool = True
    use_stratified_cv: bool = True
    use_tuning: bool = True
    use_ensembling: bool = True


active_sessions = {}
session_lock = threading.RLock()


def _cleanup_old_sessions():
    expired = []
    with session_lock:
        expired = [
            sid for sid, data in active_sessions.items()
            if data.get('status') in ("completed", "error", "stopped")
            and (time.time() - data.get('_completed_at', 0)) > SESSION_TTL
        ]
        for sid in expired:
            del active_sessions[sid]
    for sid in expired:
        _delete_session_files(sid)
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")


def _delete_file(filepath: str):
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to delete file {filepath}: {e}")


def _delete_session_files(session_id: str):
    delete_session(session_id)


def _friendly_error(raw_error: str) -> str:
    msg = str(raw_error)
    if "Column" in msg and "not found" in msg:
        return "目标列不存在，请检查配置后重试"
    if "stopped by user" in msg:
        return "训练已被用户停止"
    if "Empty DataFrame" in msg or "No data" in msg:
        return "数据为空，请检查上传的文件"
    if "target" in msg.lower() and ("missing" in msg.lower() or "not found" in msg.lower()):
        return "目标列配置有误，请返回配置页检查"
    if "MemoryError" in msg or "Cannot allocate" in msg:
        return "数据量过大导致内存不足，请减少数据行数或特征列数"
    if "Timeout" in msg:
        return "训练超时，请尝试减少模型数量或数据量"
    if len(msg) > 200:
        return "训练过程出现错误，请检查数据格式后重试"
    return msg


def _run_training_in_thread(session_id: str, params_dict: dict, model_id: str, stop_file: str):
    import subprocess
    import json
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    worker_script = os.path.join(project_root, "training_worker.py")
    cmd = ["python3", worker_script, session_id, json.dumps(params_dict)]
    log_file = f"/tmp/vibemine_train_{session_id}.log"
    
    with open(log_file, 'w') as log_f:
        subprocess.run(cmd, stdout=log_f, stderr=log_f)
    
    logger.info(f"Training subprocess finished for session {session_id}")


@router.post("/train")
async def train_models(req: TrainRequest):
    if req.task_type not in ("classification", "regression", "clustering"):
        raise HTTPException(status_code=400, detail="task_type 仅支持 classification、regression 或 clustering")

    with session_lock:
        _cleanup_old_sessions()
        if len(active_sessions) >= MAX_SESSIONS:
            raise HTTPException(status_code=429, detail="当前训练任务过多，请稍后再试")
        
        session_id = uuid.uuid4().hex[:12]
        model_id = uuid.uuid4().hex[:12]
        stop_file = f"/tmp/vibemine_stop_{session_id}"
        
        initial_data = {
            'session_id': session_id,
            'model_id': model_id,
            'status': 'training',
            'progress': 0,
            'current_model': '初始化...',
            'completed_models': [],
            'filename': req.filename,
            '_created_at': time.time()
        }
        
        init_session(session_id, initial_data)
        active_sessions[session_id] = initial_data.copy()

    params_dict = {
        "filename": req.filename,
        "task_type": req.task_type,
        "target_column": req.target_column,
        "selected_models": req.selected_models,
        "ignore_columns": req.ignore_columns,
        "use_smote": req.use_smote,
        "use_outlier_removal": req.use_outlier_removal,
        "use_advanced_imputation": req.use_advanced_imputation,
        "use_stratified_cv": req.use_stratified_cv,
        "use_tuning": req.use_tuning,
        "use_ensembling": req.use_ensembling,
        "model_id": model_id,
        "stop_file": stop_file,
    }

    thread = threading.Thread(
        target=_run_training_in_thread,
        args=(session_id, params_dict, model_id, stop_file),
        daemon=True
    )
    thread.start()

    return {"session_id": session_id, "status": "training"}


@router.get("/train/status/{session_id}")
async def get_train_status(session_id: str):
    session_data = get_session(session_id)
    if not session_data:
        with session_lock:
            cached = active_sessions.get(session_id)
        if cached:
            session_data = cached
    
    if not session_data:
        raise HTTPException(status_code=404, detail="训练会话不存在或已过期")
    
    d = {
        "session_id": session_data.get("session_id", session_id),
        "status": session_data.get("status", "training"),
        "progress": session_data.get("progress", 0),
        "current_model": session_data.get("current_model", ""),
        "completed_models": session_data.get("completed_models", []),
    }
    if session_data.get("result"):
        result_copy = dict(session_data["result"])
        result_copy.pop('shap_plot', None)
        d["result"] = result_copy
    if session_data.get("error"):
        d["error"] = session_data["error"]
    return d


@router.post("/train/stop/{session_id}")
async def stop_training(session_id: str):
    session_data = get_session(session_id)
    if not session_data:
        with session_lock:
            session_data = active_sessions.get(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="训练会话不存在或已过期")
    
    stop_file = f"/tmp/vibemine_stop_{session_id}"
    open(stop_file, 'w').close()
    
    updates = {
        'status': 'stopped',
        '_completed_at': time.time()
    }
    update_session(session_id, updates)
    
    with session_lock:
        if session_id in active_sessions:
            active_sessions[session_id].update(updates)
    
    return {"status": "stopped"}


@router.get("/train/ai-evaluation/{session_id}")
async def get_ai_evaluation(session_id: str):
    raise HTTPException(status_code=501, detail="AI evaluation 已移至前端执行，请在评估页点击 AI 按钮")


@router.get("/train/ai-misclassified/{session_id}")
async def get_ai_misclassified(session_id: str):
    raise HTTPException(status_code=501, detail="AI misclassified analysis 已移至前端执行，请在评估页点击 AI 按钮")
