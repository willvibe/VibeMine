import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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
