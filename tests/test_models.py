from core.models import Frame
import base64

def test_frame_model_validation():

    frame_bytes = b"dummy_image_data"
    frame = Frame(timestamp=1.0, embedding=[0.1, 0.2], frame=frame_bytes)
    assert frame.frame == base64.b64encode(frame_bytes).decode('utf-8')

    frame = Frame(timestamp=1.0, embedding=[0.1, 0.2])
    assert frame.frame is None
