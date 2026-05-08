import json
import os
import threading
import time
from pathlib import Path

SESSION_STATE_DIR = Path("/tmp/vibemine_sessions")
SESSION_STATE_DIR.mkdir(exist_ok=True)

_session_cache = {}
_cache_lock = threading.Lock()


def _get_path(session_id: str) -> Path:
    return SESSION_STATE_DIR / f"{session_id}.json"


def _load_from_file(session_id: str) -> dict:
    path = _get_path(session_id)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_to_file(session_id: str, data: dict):
    path = _get_path(session_id)
    with open(path, 'w') as f:
        json.dump(data, f)


def init_session(session_id: str, initial_data: dict):
    with _cache_lock:
        _session_cache[session_id] = initial_data.copy()
    _save_to_file(session_id, initial_data)


def update_session(session_id: str, updates: dict):
    with _cache_lock:
        if session_id in _session_cache:
            _session_cache[session_id].update(updates)
    _save_to_file(session_id, _session_cache.get(session_id, {}))


def get_session(session_id: str) -> dict:
    data = _load_from_file(session_id)
    if data:
        return data
    return None


def delete_session(session_id: str):
    with _cache_lock:
        _session_cache.pop(session_id, None)
    path = _get_path(session_id)
    if path.exists():
        path.unlink()


def cleanup_expired(ttl: int = 3600):
    now = time.time()
    expired = []
    with _cache_lock:
        for sid in list(_session_cache.keys()):
            data = _session_cache[sid]
            if data.get('status') in ('completed', 'error', 'stopped'):
                completed_at = data.get('_completed_at', 0)
                if (now - completed_at) > ttl:
                    expired.append(sid)
                    del _session_cache[sid]
    for sid in expired:
        path = _get_path(sid)
        if path.exists():
            path.unlink()
    return expired
