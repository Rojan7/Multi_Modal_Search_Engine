import faiss
import json
import torch
import numpy as np
from src.utils.main_utils import load_clip_model
from src.entity.artifact_entity import FaissIndexingArtifact,EmbeddingGenerationArtifact,ModelFineTuningArtifact
from PIL import Image
from src.logger import logger
from exception import MyException
import sys


class Retriever:
    def __init__(self, 
                 faiss_indexing_artifact:FaissIndexingArtifact,
                 embeddings_generation_artifact:EmbeddingGenerationArtifact,
                 model_fine_tuning_artifact: ModelFineTuningArtifact):
        self.model_path=model_fine_tuning_artifact.Model_Path
        self.model,self.processor,self.device = load_clip_model(model_path=self.model_path)
        self.mapping_path=EmbeddingGenerationArtifact.mapping_path
        self.faiss_index_path=FaissIndexingArtifact.Faiss_path

        self.index = faiss.read_index(self.faiss_index_path)

        with open(self.mapping_path) as f:
            self.mapping = json.load(f)

    def search_text(self, query, top_k=5):
        try:
            logger.info("Entered search_text method")
            inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_emb = self.model.get_text_features(**inputs)

            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb.cpu().numpy().astype("float32")

            scores, indices = self.index.search(text_emb, top_k)

            results = [
                {"url": self.mapping[str(i)], "score": float(scores[0][idx])}
                for idx, i in enumerate(indices[0])
            ]
            return results
        except Exception as e:
            raise MyException(e,sys)


    def search_image(self, image, top_k=5):
        try:
            logger.info("Entered search_image_method of Retrival class")


            # Handle different input types
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                image = Image.open(image).convert("RGB")


            inputs = self.processor(images=image, return_tensors="pt").to(self.device)


            with torch.no_grad():
                image_emb = self.model.get_image_features(**inputs)

            # Normalize (CRITICAL)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

            # Convert to numpy
            image_emb = image_emb.cpu().numpy().astype("float32")

            # Search FAISS
            scores, indices = self.index.search(image_emb, top_k)

            # Return with scores (better)
            results = [
                {"url": self.mapping[str(i)], "score": float(scores[0][idx])}
                for idx, i in enumerate(indices[0])
            ]

            return results
        
        except Exception as e:
            raise MyException(e,sys)
    def search_results(self,query):
        try:
            logger.info("Entered search result method of Retriever")
            if isinstance(query,object):
                return self.search_image(query)
            else:
                return self.search_text(query)
        except Exception as e:
            raise MyException(e,sys)