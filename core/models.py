import base64
from pydantic import BaseModel, field_validator
from typing import List, Optional

class Frame(BaseModel):
    timestamp: float
    embedding: List[float]
    frame: Optional[str] = None

    @field_validator('frame',  mode='before')
    def validate_frame(cls, v):
        if v is None: return None
        if isinstance(v, bytes):
            return base64.b64encode(v).decode('utf-8')
        return v

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
