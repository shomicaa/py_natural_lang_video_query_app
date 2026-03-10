import os
import shutil
import chromadb
import numpy as np
from typing import List
from core.models import Frame, SearchResult

class VectorDB:

    def __init__(self, collection_name: str = "video_frames"):
        self.persist_dir = "storage/chroma"

        self.cleanup()

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=chromadb.Settings(allow_reset=True)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def cleanup(self):
        if hasattr(self, 'client') and self.client:
            self.client.reset()
            del self.client
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir, ignore_errors=True)

    def add_frames(self, video_id: str, frames: List[Frame]):
        self.collection.add(
            ids=[f"{video_id}_{i}" for i in range(len(frames))],
            embeddings=[frame.embedding for frame in frames],
            metadatas=[{
                "timestamp": frame.timestamp,
                "frame": frame.frame
            } for frame in frames]
        )

    def query(self, text_embedding: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        results = self.collection.query(
            query_embeddings=[text_embedding.tolist()[0]],
            n_results=top_k,
            include=["distances", "metadatas"]
        )

        def calculate_confidence(distance):
            similarity = 1 - distance
            if similarity < 0.20: return similarity * 1.0  # 0-20% for similarities 0-0.2
            elif similarity < 0.35: return 0.20 + (similarity - 0.20) * 4.0  # 20% to 80% for 0.2-0.35
            elif similarity < 0.40: return 0.80 + (similarity - 0.35) * 4.0  # 80% to 100% for 0.35-0.4
            else: return 1.0

        confidences = [min(1.0, calculate_confidence(d)) for d in results["distances"][0]]

        return [
            SearchResult(
                timestamp=results["metadatas"][0][i]["timestamp"],
                score=confidences[i],
                id=results["ids"][0][i]
            )
            for i in range(top_k)
        ]
