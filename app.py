"""
Hugging Face Space app wrapper
This makes our FastAPI app compatible with Spaces
"""
from app.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)