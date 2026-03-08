import base64
import cv2
import numpy as np
from typing import List
from tqdm import tqdm
from core.models import Frame
from core.embedding import ClipEmbedder

class VideoProcessor:

    def __init__(self, frame_rate: int = 3):
        self.frame_rate = frame_rate
        self.embedder = ClipEmbedder()

    def process_video(self, video_path: str) -> List[Frame]:
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.frame_rate))

        try:
            with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Video", unit="frame") as progress_bar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    progress_bar.update(1)
                    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
                        frames.append(self._process_frame(frame, cap.get(cv2.CAP_PROP_POS_FRAMES)/fps))
        finally:
            cap.release()

        return frames

    def _process_frame(self, frame: np.ndarray, timestamp: float) -> Frame:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding = self.embedder.encode_image(rgb_frame)

        success, buffer = cv2.imencode(
            '.jpg',
            cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )

        if not success:
            raise ValueError("JPEG encoding failed")

        return Frame(
            timestamp=timestamp,
            embedding=embedding.tolist()[0],
            frame=base64.b64encode(buffer).decode('utf-8')
        )
