import torch
import numpy as np
from src.entity.config_entity import embeddings_generations_config
from src.entity.artifact_entity import DataExtractorArtifact,ModelFineTuningArtifact,EmbeddingGenerationArtifact
import json
import os
import sys
from src.exception import MyException
from src.logger import logger
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from pathlib import Path


class EmbeddingGeneration:
    def __init__(self,
                 embeddings_genrations_config:embeddings_generations_config,
                 data_extractor_artifact:DataExtractorArtifact,
                 model_fine_tuning_artifact:ModelFineTuningArtifact):
        self.embeddings_generation_config=embeddings_generations_config
        self.data_extractor_artifact=data_extractor_artifact
        self.model_fine_tuning_artifact=model_fine_tuning_artifact
        self.device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model=CLIPModel.from_pretrained(self.model_fine_tuning_artifact.Model_Path).to(self.device)
        self.processor=CLIPProcessor.from_pretrained(self.model_fine_tuning_artifact.Model_Path)
        self.model.eval()
        
    def generate_embeddings(self):
        try:
            logger.info("Embedding generation started")

            image_embeds_list = []
            text_embeds_list = []
            mapping = {}

            for idx, ((img_path, caption_path), page_url) in enumerate(
                self.data_extractor_artifact.Img_to_url.items()
            ):
                image = Image.open(img_path).convert("RGB")

                with open(caption_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                inputs = self.processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"]
                    )

                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds

                    # Normalize
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                image_embeds_list.append(image_embeds.cpu().numpy())
                text_embeds_list.append(text_embeds.cpu().numpy())

                mapping[idx] = page_url

            # Stack
            image_embeds = np.vstack(image_embeds_list)
            text_embeds = np.vstack(text_embeds_list)
        

            # Save separately
            save_path=Path(self.embeddings_generation_config.embeddings_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, image_embeds)
            np.save(save_path,text_embeds)

            return mapping,save_path

        except Exception as e:
            raise MyException(e, sys)
        
    def save_mappings(self,mapping):
        try:
            logger.info("Savign mappings")
            path=self.embeddings_generation_config.Mapping_path
            with open(path,"w") as f:
                json.dump(mapping,f)
                
            return path
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_embeddings_generations(self):
        try:
            logger.info("Intiating Embedding Generations")
            mappings,embeddings_path=self.generate_embeddings()
            mapping_path=self.save_mappings(mappings)
            
            return EmbeddingGenerationArtifact(
                embeddings_path,
                "Successfull",
                mapping_path
                
                
            )
        except Exception as e:
            raise MyException(e,sys)
            

            
            

            
            
            
    