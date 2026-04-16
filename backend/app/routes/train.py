import os
import threading
import uuid
import logging
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from app.services.train_service import run_automl
from app.services.ai_service import get_model_evaluation, get_misclassified_analysis
from app.config import UPLOAD_DIR, MODEL_DIR

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


training_sessions = {}
session_lock = threading.Lock()


def _cleanup_old_sessions():
    now = time.time()
    expired = [
        sid
        for sid, s in training_sessions.items()
        if s.status in ("completed", "error", "stopped")
        and hasattr(s, "_completed_at")
        and (now - s._completed_at) > SESSION_TTL
    ]
    for sid in expired:
        session = training_sessions[sid]
        session.cleanup()
        del training_sessions[sid]
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")


def _delete_file(filepath: str):
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to delete file {filepath}: {e}")


class TrainingSession:
    def __init__(self, params: TrainRequest):
        self.session_id = uuid.uuid4().hex[:12]
        self.params = params
        self.model_id = None
        self.status = "training"
        self.progress = 0
        self.current_model = "初始化..."
        self.completed_models = []
        self.stop_event = threading.Event()
        self.result = None
        self.error = None
        self._completed_at = None

    def to_dict(self):
        d = {
            "session_id": self.session_id,
            "status": self.status,
            "progress": self.progress,
            "current_model": self.current_model,
            "completed_models": self.completed_models,
        }
        if self.result:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d

    def cleanup(self, delete_model=True):
        csv_path = os.path.join(UPLOAD_DIR, self.params.filename)
        _delete_file(csv_path)
        if delete_model and self.model_id:
            model_path = os.path.join(MODEL_DIR, f"{self.model_id}.pkl")
            _delete_file(model_path)


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


def _progress_cb(session: TrainingSession):
    def callback(progress: int, message: str = "", completed_models: list = None):
        session.progress = progress
        session.current_model = message
        if completed_models:
            session.completed_models = completed_models
        return session.stop_event.is_set()
    return callback


def _run_training(session: TrainingSession):
    session.model_id = uuid.uuid4().hex[:12]
    try:
        result = run_automl(
            filename=session.params.filename,
            task_type=session.params.task_type,
            target_column=session.params.target_column,
            selected_models=session.params.selected_models,
            ignore_columns=session.params.ignore_columns,
            use_smote=session.params.use_smote,
            use_outlier_removal=session.params.use_outlier_removal,
            use_advanced_imputation=session.params.use_advanced_imputation,
            use_stratified_cv=session.params.use_stratified_cv,
            use_tuning=session.params.use_tuning,
            use_ensembling=session.params.use_ensembling,
            progress_callback=_progress_cb(session),
            stop_event=session.stop_event,
            model_id=session.model_id,
        )

        session.result = result
        session.status = "completed"
        session.progress = 100
        session.current_model = ""
        session._completed_at = time.time()
        session.cleanup(delete_model=False)
        logger.info(f"Training session {session.session_id} completed successfully")

    except Exception as e:
        if session.stop_event.is_set():
            session.error = "训练已被用户停止"
            session.status = "stopped"
            session._completed_at = time.time()
            logger.info(f"Training session {session.session_id} stopped by user")
        else:
            friendly = _friendly_error(str(e))
            session.error = friendly
            session.status = "error"
            session._completed_at = time.time()
            logger.error(f"Training session {session.session_id} failed: {str(e)}", exc_info=True)
        session.cleanup()


@router.post("/train")
async def train_models(req: TrainRequest):
    if req.task_type not in ("classification", "regression", "clustering"):
        raise HTTPException(status_code=400, detail="task_type 仅支持 classification、regression 或 clustering")

    with session_lock:
        _cleanup_old_sessions()
        if len(training_sessions) >= MAX_SESSIONS:
            raise HTTPException(status_code=429, detail="当前训练任务过多，请稍后再试")
        session = TrainingSession(req)
        training_sessions[session.session_id] = session

    thread = threading.Thread(target=_run_training, args=(session,), daemon=True)
    thread.start()

    return {"session_id": session.session_id, "status": "training"}


@router.get("/train/status/{session_id}")
async def get_train_status(session_id: str):
    with session_lock:
        session = training_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="训练会话不存在或已过期")
    d = {
        "session_id": session.session_id,
        "status": session.status,
        "progress": session.progress,
        "current_model": session.current_model,
        "completed_models": session.completed_models,
    }
    if session.result:
        result_copy = dict(session.result)
        result_copy.pop('shap_plot', None)
        d["result"] = result_copy
    if session.error:
        d["error"] = session.error
    return d


@router.post("/train/stop/{session_id}")
async def stop_training(session_id: str):
    with session_lock:
        session = training_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="训练会话不存在或已过期")
    session.stop_event.set()
    with session_lock:
        session.status = "stopped"
        session._completed_at = time.time()
    return {"status": "stopped"}


@router.get("/train/ai-evaluation/{session_id}")
async def get_ai_evaluation(session_id: str, x_api_key: str = Header(None)):
    with session_lock:
        session = training_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="训练会话不存在或已过期")
    if session.result is None:
        raise HTTPException(status_code=400, detail="训练结果不存在")
    if "ai_evaluation" in session.result and session.result["ai_evaluation"]:
        return {"ai_evaluation": session.result["ai_evaluation"]}
    try:
        ai_evaluation = get_model_evaluation(
            session.result["metrics_table"],
            session.params.target_column,
            session.params.task_type,
            api_key=x_api_key,
        )
    except Exception as e:
        logger.error(f"AI evaluation failed: {e}", exc_info=True)
        ai_evaluation = "**AI 评估暂时不可用**\n\n请稍后重试或检查 API Key 配置"
    with session_lock:
        session.result["ai_evaluation"] = ai_evaluation
    return {"ai_evaluation": ai_evaluation}


@router.get("/train/ai-misclassified/{session_id}")
async def get_ai_misclassified(session_id: str, x_api_key: str = Header(None)):
    with session_lock:
        session = training_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="训练会话不存在或已过期")
    if session.result is None:
        raise HTTPException(status_code=400, detail="训练结果不存在")
    if "misclassified_analysis" in session.result and session.result["misclassified_analysis"]:
        return {"misclassified_analysis": session.result["misclassified_analysis"]}
    try:
        misclassified_analysis = get_misclassified_analysis(
            session.result.get("misclassified_samples", []),
            session.params.target_column,
            session.result.get("feature_importance", {}),
            api_key=x_api_key,
        )
    except Exception as e:
        logger.error(f"AI misclassified analysis failed: {e}", exc_info=True)
        misclassified_analysis = "**错误样本分析暂时不可用**\n\n请稍后重试或检查 API Key 配置"
    with session_lock:
        session.result["misclassified_analysis"] = misclassified_analysis
    return {"misclassified_analysis": misclassified_analysis}
