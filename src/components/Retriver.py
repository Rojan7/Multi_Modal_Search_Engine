import faiss
import json
import torch
import numpy as np
from src.utils.main_utils import load_clip_model
from src.entity.artifact_entity import FaissIndexingArtifact,EmbeddingGenerationArtifact,ModelFineTuningArtifact
from PIL import Image
from src.logger import logger
from src.exception import MyException
import sys
import os

class Retriever:
    def __init__(self, Faiss_path, mapping_path, Model_Path):

        self.model_path = Model_Path
        self.model, self.processor, self.device = load_clip_model(self.model_path)

        self.mapping_path = mapping_path
        self.faiss_path = Faiss_path

        # FIX 1
        self.index = faiss.read_index(self.faiss_path)

        with open(self.mapping_path) as f:
            self.mapping = json.load(f)

    def predict(self, query, top_k=5):
        try:
            logger.info("Entered predict method")

            if isinstance(query, str) and not os.path.exists(query):
                # TEXT QUERY
                inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)

                with torch.no_grad():
                    emb = self.model.get_text_features(**inputs)

            else:
                # IMAGE QUERY
                if isinstance(query, str):
                    image = Image.open(query).convert("RGB")
                elif isinstance(query, Image.Image):
                    image = query.convert("RGB")
                else:
                    image = Image.open(query).convert("RGB")

                inputs = self.processor(images=image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    emb = self.model.get_image_features(**inputs)

            # Normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)

            # FAISS search
            emb = emb.cpu().numpy().astype("float32")
            scores, indices = self.index.search(emb, top_k)

            # Results
            results = [
                {
                    "url": self.mapping.get(str(i), self.mapping.get(i)),
                    "score": float(scores[0][idx])
                }
                for idx, i in enumerate(indices[0])
            ]

            logger.info("Results returned successfully")
            return results

        except Exception as e:
            raise MyException(e, sys)