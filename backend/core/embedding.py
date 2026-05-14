import torch
from transformers import CLIPModel, CLIPProcessor
import numpy as np

class ClipEmbedder:

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()
