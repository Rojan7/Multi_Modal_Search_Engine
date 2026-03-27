import faiss
import json
import torch
import numpy as np


class Retriever:
    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

        self.index = faiss.read_index(config.faiss_index_path)

        with open(config.mapping_path) as f:
            self.mapping = json.load(f)

    def search_text(self, query, top_k=5):
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)

        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb.cpu().numpy().astype("float32")

        scores, indices = self.index.search(text_emb, top_k)

        results = [self.mapping[str(i)] for i in indices[0]]
        return results