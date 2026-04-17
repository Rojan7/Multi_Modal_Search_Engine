import faiss
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import boto3
import io
import os
from src.logger import logger
from src.exception import MyException
import tempfile
import sys

class Retriever:
    def __init__(self, s3_bucket, model_folder_key, faiss_key, mapping_key):
        """
        s3_bucket: str -> S3 bucket name
        model_folder_key: str -> folder path of fine-tuned CLIP model in S3 (contains config.json, model.safetensors, processor_config.json)
        faiss_key: str -> S3 path to FAISS index (.index file)
        mapping_key: str -> S3 path to mapping JSON
        """
        try:
            logger.info("Initializing Retriever from S3")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load CLIP model + processor
            self.model, self.processor = self._load_clip_model_s3(s3_bucket, model_folder_key)

            # Load FAISS index
            self.index = self._load_faiss_index_s3(s3_bucket, faiss_key)

            # Load mapping JSON
            self.mapping = self._load_mapping_s3(s3_bucket, mapping_key)

            logger.info("Retriever initialized successfully")

        except Exception as e:
            raise MyException(e, sys)

    def _load_clip_model_s3(self, s3_bucket, model_folder_key):
        """
        Downloads model folder from S3 and loads CLIP model + processor
        """
        s3 = boto3.client('s3')
        with tempfile.TemporaryDirectory() as tmp_dir:
            # download each file in the folder
            # Here we assume the model folder contains: config.json, processor_config.json, model.safetensors
            files_to_download = ["config.json", "processor_config.json", "model.safetensors", 
                                 "tokenizer.json", "tokenizer_config.json"]
            for f in files_to_download:
                s3.download_file(s3_bucket, f"{model_folder_key}{f}", os.path.join(tmp_dir, f))

            # Load model
            model = CLIPModel.from_pretrained(tmp_dir).to(self.device)
            processor = CLIPProcessor.from_pretrained(tmp_dir)
            model.eval()
        return model, processor

    def _load_faiss_index_s3(self, s3_bucket, faiss_key):
        """
        Downloads FAISS index from S3 into memory
        """
        s3 = boto3.client('s3')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp_file:
            s3.download_file(s3_bucket, faiss_key, tmp_file.name)
            index = faiss.read_index(tmp_file.name)
            os.unlink(tmp_file.name)
        return index

    def _load_mapping_s3(self, s3_bucket, mapping_key):
        """
        Downloads mapping JSON from S3
        """
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=s3_bucket, Key=mapping_key)
        mapping = json.loads(obj['Body'].read())
        return mapping

    def predict(self, query, top_k=5):
        """
        query: str or PIL.Image.Image
        top_k: number of results
        """
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

            # Map results
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