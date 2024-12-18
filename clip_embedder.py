from PIL import Image
from typing import List
import logging
import torch
import clip
import torch.nn.functional as F

# Set up logging
logger = logging.getLogger(__name__)

class CLIPEmbedder:
    def __init__(self, model_name: str = "ViT-B/32") -> None:
        logger.info("Initializing CLIPEmbedder")
        self.model = None
        self.preprocess = None

        try:
            # Check for MPS first, then CUDA, then fall back to CPU
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            logger.info(f"Using device: {self.device}")

            logger.info(f"Loading model: {model_name}")
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def get_image_embeddings(self, images: List[Image.Image]):
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            # Preprocess images and create batch
            processed_images = torch.cat([
                self.preprocess(image).unsqueeze(0).to(self.device)
                for image in images
            ])

            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(processed_images)
                # Normalize embeddings
                image_features = F.normalize(image_features, dim=-1)

            return image_features.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error processing images: {str(e)}")
            raise RuntimeError(f"Error processing images: {str(e)}")

    def get_text_embeddings(self, texts: List[str]):
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            # Tokenize and encode text
            text_tokens = clip.tokenize(texts).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                # Normalize embeddings
                text_features = F.normalize(text_features, dim=-1)

            return text_features.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise RuntimeError(f"Error encoding text: {str(e)}")

    def compute_similarity(self, embeddings1, embeddings2):
        try:
            emb1 = torch.tensor(embeddings1)
            emb2 = torch.tensor(embeddings2)
            # Computing cosine similarity
            similarities = torch.matmul(emb1, emb2.T)
            return similarities.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise RuntimeError(f"Error computing similarity: {str(e)}")