from fastapi.testclient import TestClient
from api import app
import pytest

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):

    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_upload_video(client, mocker):

    mock_process = mocker.patch("core.processor.VideoProcessor.process_video", return_value=[])
    mock_add = mocker.patch("core.database.VectorDB.add_frames")

    test_file = ("test.mp4", b"dummy_video_content", "video/mp4")
    response = client.post("/upload", files={"file": test_file})

    assert response.status_code == 200
    mock_process.assert_called_once()
    mock_add.assert_called_once_with("test.mp4", [])
