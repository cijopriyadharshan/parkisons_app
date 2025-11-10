# run.py
from app import app  # Flask app
from threading import Thread
import uvicorn

def run_fastapi():
    uvicorn.run("app:fastapi_app", host="0.0.0.0", port=8000, log_level="error")

if __name__ == "__main__":
    Thread(target=run_fastapi, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
