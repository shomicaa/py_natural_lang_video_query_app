import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.processor import VideoProcessor
from core.database import VectorDB
from core.models import VideoUploadResponse, SearchResponse, SearchRequest

app = FastAPI(title="Video Search API")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = VideoProcessor()
db = VectorDB()

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Video Search API is running"}

@app.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    video_path = f"storage/temp_{file.filename}"
    try:
        os.makedirs("storage", exist_ok=True)
        with open(video_path, "wb") as f:
            while contents := await file.read(1024 * 1024): f.write(contents)

        frames = processor.process_video(video_path)
        db.add_frames(file.filename, frames)
        return VideoUploadResponse(status="success", frames_processed=len(frames))

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail="Video processing failed. Please try a different file.")
    finally:
        if os.path.exists(video_path): os.remove(video_path)

@app.post("/search", response_model=SearchResponse)
async def search_video(request: SearchRequest):
    try:
        embedding = processor.embedder.encode_text(request.prompt)
        return SearchResponse(results=db.query(embedding))
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(500, "Search failed")
