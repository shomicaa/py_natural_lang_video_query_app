import base64
import os
import logging
from fastapi import FastAPI, Response, UploadFile, File, HTTPException
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

@app.post("/cleanup")
async def cleanup_database():
    try:
        db.cleanup()
        return {"status": "success", "message": "Database completely reset"}
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

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

@app.get("/frame/{frame_id}")
async def get_frame(frame_id: str):
    from urllib.parse import unquote
    frame_id = unquote(frame_id)

    result = db.collection.get(ids=[frame_id], include=["metadatas"])
    if not result or not result['metadatas'] or not result['metadatas'][0]:
        raise HTTPException(404, detail="Frame metadata not found")

    frame_data = result['metadatas'][0].get('frame')
    if not frame_data: raise HTTPException(404, detail="Frame image data not available")

    try:
        return Response(content=base64.b64decode(frame_data), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Frame decoding failed: {str(e)}")
        raise HTTPException(422, detail="Invalid frame data format")
