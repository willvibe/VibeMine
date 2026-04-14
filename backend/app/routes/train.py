import threading
import uuid
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.train_service import run_automl
from app.services.ai_service import get_model_evaluation

router = APIRouter(prefix="/api", tags=["train"])
logger = logging.getLogger(__name__)


class TrainRequest(BaseModel):
    filename: str
    task_type: str
    target_column: str
    selected_models: list = None
    ignore_columns: list = None
    use_smote: bool = False


training_sessions = {}
session_lock = threading.Lock()


class TrainingSession:
    def __init__(self, params: TrainRequest):
        self.session_id = uuid.uuid4().hex[:12]
        self.params = params
        self.status = "training"
        self.progress = 0
        self.current_model = "初始化..."
        self.completed_models = []
        self.stop_event = threading.Event()
        self.result = None
        self.error = None

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


def _progress_cb(session: TrainingSession):
    def callback(progress: int, message: str = "", completed_models: list = None):
        session.progress = progress
        session.current_model = message
        if completed_models:
            session.completed_models = completed_models
        return session.stop_event.is_set()
    return callback


def _run_training(session: TrainingSession):
    try:
        result = run_automl(
            filename=session.params.filename,
            task_type=session.params.task_type,
            target_column=session.params.target_column,
            selected_models=session.params.selected_models,
            ignore_columns=session.params.ignore_columns,
            use_smote=session.params.use_smote,
            progress_callback=_progress_cb(session),
            stop_event=session.stop_event,
        )

        ai_evaluation = ""
        try:
            ai_evaluation = get_model_evaluation(
                result["metrics_table"],
                session.params.target_column,
                session.params.task_type,
            )
        except Exception as e:
            ai_evaluation = f"**AI 评估暂时不可用**\n\n错误信息：{str(e)}"

        result["ai_evaluation"] = ai_evaluation
        session.result = result
        session.status = "completed"
        session.progress = 100
        session.current_model = ""
        logger.info(f"Training session {session.session_id} completed. Result: {result}")
    except Exception as e:
        session.error = str(e)
        session.status = "error"


@router.post("/train")
async def train_models(req: TrainRequest):
    if req.task_type not in ("classification", "regression", "clustering"):
        raise HTTPException(status_code=400, detail="task_type 仅支持 classification、regression 或 clustering")

    session = TrainingSession(req)
    with session_lock:
        training_sessions[session.session_id] = session

    thread = threading.Thread(target=_run_training, args=(session,), daemon=True)
    thread.start()

    return {"session_id": session.session_id, "status": "training"}


@router.get("/train/status/{session_id}")
async def get_train_status(session_id: str):
    with session_lock:
        session = training_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.post("/train/stop/{session_id}")
async def stop_training(session_id: str):
    with session_lock:
        session = training_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.stop_event.set()
    session.status = "stopped"
    return {"status": "stopped"}