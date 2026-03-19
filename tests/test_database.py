from core.database import VectorDB
from core.models import Frame
import numpy as np

def test_add_and_query_frames():
    db = VectorDB(collection_name="test_collection")
    frames = [
        Frame(timestamp=1.0, embedding=[0.1, 0.2], frame="dummy_base64"),
        Frame(timestamp=2.0, embedding=[0.3, 0.4], frame="dummy_base64")
    ]

    db.add_frames("test_video", frames)
    assert db.collection.count() == 2

    results = db.query(np.array([[0.1, 0.2]]), top_k=1)
    assert len(results) == 1
    assert results[0].timestamp == 1.0

    db.client.delete_collection("test_collection")
