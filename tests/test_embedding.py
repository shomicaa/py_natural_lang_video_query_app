import numpy as np
import pytest
from core.embedding import ClipEmbedder

TEST_MODEL = "openai/clip-vit-base-patch32"

@pytest.fixture
def embedder():
    return ClipEmbedder(model_name=TEST_MODEL)

def test_image_embedding(embedder):

    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

    embedding = embedder.encode_image(dummy_image)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 512)
    assert np.all(np.isfinite(embedding))

def test_text_embedding(embedder):

    embedding = embedder.encode_text("a cat sitting on a mat")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 512)
    assert np.all(np.isfinite(embedding))
