# run.py
from app import fastapi_app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:fastapi_app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        timeout_keep_alive=180
    )
