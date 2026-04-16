import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 代理配置（国内服务器需配置代理才能访问 Gemini API）
GEMINI_PROXY = os.getenv("GEMINI_PROXY", "")
if GEMINI_PROXY:
    os.environ["HTTPS_PROXY"] = GEMINI_PROXY
    os.environ["https_proxy"] = GEMINI_PROXY

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://vibemine.pages.dev",
    "http://45.113.2.167:5173",
    "http://45.113.2.167:3000",
    "http://45.113.2.167:4173",
    "http://45.113.2.167:5000",
    "http://45.113.2.167",
]
