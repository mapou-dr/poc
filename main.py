from ray import serve
from PIL import Image
import io
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from starlette.responses import JSONResponse
import logging
import torch
import clip
import torch.nn.functional as F
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextRequest(BaseModel):
    texts: List[str]

# Rest of the CLIPEmbedder class remains the same
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


@serve.deployment
@serve.ingress(app)
class EmbeddingDeployment:
    def __init__(self):
        logger.info("Initializing EmbeddingDeployment")
        try:
            self.generator = CLIPEmbedder()
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingDeployment: {str(e)}")
            raise

    @app.post("/embeddings/images")
    async def create_image_embeddings(self, files: List[UploadFile]):
        if not files:
            raise HTTPException(status_code=400, detail="No images provided")

        try:
            images = []
            filenames = []

            for file in files:
                content = await file.read()
                try:
                    img = Image.open(io.BytesIO(content))
                    images.append(img)
                    filenames.append(file.filename)
                except Exception as e:
                    logger.error(f"Error processing image {file.filename}: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing image {file.filename}: {str(e)}"
                    )

            embeddings = self.generator.get_image_embeddings(images)

            result = {
                "embeddings": [
                    {
                        "filename": filename,
                        "embedding": embedding
                    }
                    for filename, embedding in zip(filenames, embeddings)
                ]
            }

            return JSONResponse(content=result)

        except Exception as e:
            logger.error(f"Error in create_image_embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/embeddings/text")
    async def create_text_embeddings(self, request: TextRequest):
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        try:
            embeddings = self.generator.get_text_embeddings(request.texts)

            result = {
                "embeddings": [
                    {
                        "text": text,
                        "embedding": embedding
                    }
                    for text, embedding in zip(request.texts, embeddings)
                ]
            }

            return JSONResponse(content=result)

        except Exception as e:
            logger.error(f"Error in create_text_embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/similarity")
    async def compute_similarity(
            self,
            files: List[UploadFile] = File(...),
            text_request: str = Form(...)
    ):
        try:
            # Parse the text_request JSON string
            try:
                text_data = json.loads(text_request)
                text_request_obj = TextRequest(**text_data)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail="Invalid JSON in text_request")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid text_request format: {str(e)}")

            # Get image embeddings
            images = [Image.open(io.BytesIO(await file.read())) for file in files]
            image_embeddings = self.generator.get_image_embeddings(images)

            # Get text embeddings
            text_embeddings = self.generator.get_text_embeddings(text_request_obj.texts)

            # Compute similarities
            similarities = self.generator.compute_similarity(image_embeddings, text_embeddings)

            result = {
                "similarities": [
                    {
                        "image": file.filename,
                        "scores": [
                            {
                                "text": text,
                                "score": float(score)
                            }
                            for text, score in zip(text_request_obj.texts, sim_scores)
                        ]
                    }
                    for file, sim_scores in zip(files, similarities)
                ]
            }

            return JSONResponse(content=result)

        except Exception as e:
            logger.error(f"Error in compute_similarity: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check(self):
        try:
            if self.generator.model is None:
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "detail": "Model not initialized"}
                )
            return {"status": "healthy"}
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "detail": str(e)}
            )


app = EmbeddingDeployment.bind()