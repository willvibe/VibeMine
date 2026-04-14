import asyncio
import json
import uuid
import logging
from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.train_service import run_automl_async
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["websocket"])


class TrainingSession:
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.status = "idle"
        self.progress = 0
        self.current_model = ""
        self.completed_models = []
        self.metrics_table = []
        self.feature_importance = {}
        self.ai_evaluation = ""
        self.error = None
        self.stop_event = threading.Event()
        self._result = None
        self._error = None
        self._update_event = threading.Event()

    def to_dict(self):
        return {
            "session_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "current_model": self.current_model,
            "completed_models": self.completed_models,
            "total_models": len(self.completed_models) + (1 if self.current_model else 0),
        }

    def has_update(self) -> bool:
        if self._update_event.is_set():
            self._update_event.clear()
            return True
        return False


sessions: Dict[str, TrainingSession] = {}
session_lock = threading.Lock()


async def progress_callback(session_id: str, progress: int, msg: str = "", completed: list = None):
    with session_lock:
        if session_id in sessions:
            session = sessions[session_id]
            session.progress = progress
            session.current_model = msg
            if completed:
                session.completed_models = completed
            session._update_event.set()
            logger.info(f"Progress: {progress}%, {msg}")
            return session.stop_event.is_set()
    return True


@router.websocket("/ws/train/{client_id}")
async def websocket_train(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connected: {client_id}")

    session_id = client_id
    with session_lock:
        sessions[session_id] = TrainingSession()
        session = sessions[session_id]

    update_task = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"Received message: {message.get('action')}")

            if message.get("action") == "start":
                session.status = "training"
                session.stop_event.clear()
                session.progress = 0
                session.completed_models = []
                session.error = None
                session.current_model = "初始化..."
                session._result = None
                session._error = None

                await websocket.send_json({
                    "type": "status",
                    "status": "training",
                    "progress": 0,
                    "message": "开始训练..."
                })

                async def send_updates():
                    while True:
                        with session_lock:
                            session_ref = sessions.get(session_id)
                            if not session_ref:
                                break

                            if session_ref._result:
                                try:
                                    await websocket.send_json({
                                        "type": "result",
                                        "status": "completed",
                                        "progress": 100,
                                        "result": session_ref._result
                                    })
                                    logger.info("Sent final result to client")
                                except Exception as e:
                                    logger.error(f"Error sending result: {e}")
                                break

                            if session_ref._error:
                                try:
                                    await websocket.send_json({
                                        "type": "error",
                                        "status": "error",
                                        "message": session_ref._error
                                    })
                                except:
                                    pass
                                break

                            if session_ref.has_update() or session_ref.status == "training":
                                try:
                                    await websocket.send_json({
                                        "type": "status",
                                        "status": session_ref.status,
                                        "progress": session_ref.progress,
                                        "message": session_ref.current_model,
                                        "current_model": session_ref.current_model,
                                        "completed_models": session_ref.completed_models,
                                    })
                                except Exception as e:
                                    logger.error(f"Error sending update: {e}")
                                    break

                        await asyncio.sleep(0.2)

                if update_task:
                    update_task.cancel()
                update_task = asyncio.create_task(send_updates())

                params = message.get("params", {})
                task_type = params.get("task_type", "classification")
                target_column = params.get("target_column", "")
                logger.info(f"Params: task_type={task_type}, target={target_column}")

                def run_training():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    async def train_with_callback():
                        return await run_automl_async(
                            filename=params.get("filename"),
                            task_type=task_type,
                            target_column=target_column,
                            selected_models=params.get("selected_models"),
                            ignore_columns=params.get("ignore_columns"),
                            use_smote=params.get("use_smote"),
                            progress_callback=lambda p, m, c=None: progress_callback(session_id, p, m, c),
                            stop_event=session.stop_event,
                        )

                    try:
                        logger.info("Starting training in thread")
                        result = loop.run_until_complete(train_with_callback())
                        logger.info(f"Training completed, result keys: {result.keys()}")

                        from app.services.ai_service import get_model_evaluation
                        try:
                            ai_evaluation = get_model_evaluation(
                                result.get("metrics_table", []),
                                target_column,
                                task_type,
                            )
                        except Exception as e:
                            logger.error(f"AI evaluation error: {e}")
                            ai_evaluation = f"**AI 评估暂时不可用**\n\n错误信息：{str(e)}"

                        result["ai_evaluation"] = ai_evaluation

                        with session_lock:
                            if session_id in sessions:
                                sessions[session_id]._result = result
                                sessions[session_id].status = "completed"
                                sessions[session_id].progress = 100
                                sessions[session_id].current_model = ""
                                sessions[session_id]._update_event.set()
                        logger.info("Training fully completed, result set")
                    except Exception as e:
                        logger.error(f"Training error: {e}")
                        import traceback
                        traceback.print_exc()
                        with session_lock:
                            if session_id in sessions:
                                sessions[session_id]._error = str(e)
                                sessions[session_id].error = str(e)
                                sessions[session_id].status = "error"
                                sessions[session_id]._update_event.set()
                    finally:
                        loop.close()

                thread = threading.Thread(target=run_training, daemon=True)
                thread.start()
                logger.info("Training thread started")

            elif message.get("action") == "stop":
                session.stop_event.set()
                if update_task:
                    update_task.cancel()
                await websocket.send_json({
                    "type": "status",
                    "status": "stopped",
                    "message": "训练已停止"
                })

            elif message.get("action") == "poll":
                with session_lock:
                    if session._result:
                        if update_task:
                            update_task.cancel()
                        await websocket.send_json({
                            "type": "result",
                            "status": "completed",
                            "progress": 100,
                            "result": session._result
                        })
                        logger.info("Sent result to client (poll)")
                    elif session._error:
                        if update_task:
                            update_task.cancel()
                        await websocket.send_json({
                            "type": "error",
                            "status": "error",
                            "message": session._error
                        })
                    else:
                        await websocket.send_json({
                            "type": "status",
                            "status": session.status,
                            "progress": session.progress,
                            "message": session.current_model,
                            "current_model": session.current_model,
                            "completed_models": session.completed_models,
                        })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        session.stop_event.set()
        if update_task:
            update_task.cancel()
        with session_lock:
            if session_id in sessions:
                del sessions[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        session.status = "error"
        session.error = str(e)


@router.get("/ws/status/{session_id}")
async def get_session_status(session_id: str):
    with session_lock:
        if session_id in sessions:
            return sessions[session_id].to_dict()
    return {"status": "not_found"}


@router.post("/ws/stop/{session_id}")
async def stop_training(session_id: str):
    with session_lock:
        if session_id in sessions:
            sessions[session_id].stop_event.set()
            return {"status": "stopped"}
    return {"status": "not_found"}