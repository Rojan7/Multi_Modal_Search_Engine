import os
from src.exception import MyException
from src.logger import logger
import faiss
import numpy as np
import json
import os
from src.entity.config_entity import Faiss_config
from src.entity.artifact_entity import FaissIndexingArtifact,EmbeddingGenerationArtifact
import sys
from pathlib import Path


class FAISSIndexBuilder:
    def __init__(self,
                 embeddings_generation_artifact: EmbeddingGenerationArtifact):
        self.faiss_config = Faiss_config
        self.embeddings_generation_artifact=embeddings_generation_artifact

    def build_index(self):
        try:
            logger.info("Entered the build_index method of class FAISSIndexBuilder")
            path=self.embeddings_generation_artifact.Embeddings_path
            if path.suffix != ".npy":
                path = path.with_suffix(".npy")
            embeddings = np.load(path)
            embeddings = embeddings.astype("float32")
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            # Add embeddings
            index.add(embeddings)
            faiss.write_index(index, self.faiss_config.index_path)
            return self.faiss_config.index_path
            
        
        except Exception as e:
            raise MyException(e,sys)
        
    def initate_Faiss_index_building(self):
        try:
            logger.info("started index generation")
            index_path=self.build_index()
            return FaissIndexingArtifact(
                Faiss_path=index_path,
                status=True
            )
            logger.info("Index are succefully saved")
  
        except Exception as e:
            raise MyException(e,sys)
