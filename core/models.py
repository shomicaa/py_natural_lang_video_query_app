from pydantic import BaseModel
from typing import List, Optional

class Frame(BaseModel):
    timestamp: float
    embedding: List[float]
    frame: Optional[str] = None

class SearchResult(BaseModel):
    timestamp: float
    score: float
    id: str

class VideoUploadResponse(BaseModel):
    status: str
    frames_processed: int

class SearchResponse(BaseModel):
    results: List[SearchResult]

class SearchRequest(BaseModel):
    prompt: str
