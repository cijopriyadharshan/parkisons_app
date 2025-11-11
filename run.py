# run.py
import os
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        import subprocess
        subprocess.run(["gunicorn", "--version"], check=True, capture_output=True)
        import sys
        sys.argv = ["gunicorn", "-w", "2", "-b", f"0.0.0.0:{port}", "app:app"]
        import gunicorn.app.wsgiapp
        gunicorn.app.wsgiapp.run()
    except:
        app.run(host="0.0.0.0", port=port)
