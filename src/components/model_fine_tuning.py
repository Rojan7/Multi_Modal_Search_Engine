import torch
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
from tqdm import tqdm
import sys
import os
from src.constants import epochs,lr,batch_size
from src.logger import logger
from src.exception import MyException
from src.components.model_loader import CLIPLOADER
from src.entity.config_entity import model_fine_tuning_config
from src.entity.artifact_entity import ModelLoaderArtifact,DataExtractorArtifact

class ImageCaptionDataset(Dataset):
    def __init__(self,data_list,processor, img_size=(224,224)):
        self.data=data_list
        self.processor=processor
        self.img_size=img_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        image_path,caption_path=self.data[idx]
        image=Image.open(image_path).convert("RGB").resize(self.img_size)   
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return {"image": image, "caption": caption}
    
class CLIPFineTuner:
    def __init__(self,model_fine_tuner_config:model_fine_tuning_config,
                 model_loader_artifact:ModelLoaderArtifact,
                 data_extract_artifact:DataExtractorArtifact):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu" )
        self.model=CLIPModel.from_pretrained(ModelLoaderArtifact.LoadedModelPath).to(self.device)
        self.processor=CLIPProcessor.from_pretrained(ModelLoaderArtifact.LoadedModelPath)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
       # self.loss_fn = nn.CosineEmbeddingLoss()  # CLIP uses cosine similarity
        self.model_fine_tuner_config=model_fine_tuning_config
        self.model_loader_artifact=model_loader_artifact
        self.data_extract_artifact=data_extract_artifact
        
    def collate_fn(self,batch, processor):
        logger.info("Starting collate_fn of CLIPFineTuner class")
        try:
            images = [item["image"] for item in batch]
            captions = [item["caption"] for item in batch]

            inputs = processor(
                text=list(captions),
                images=list(images),
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            return inputs
        except Exception as e:
            raise MyException(e,sys)
    

    def load_image_caption_pairs(self):
        pairs = []
        image_dir=self.data_extract_artifact.image_dir
        caption_dir=self.data_extract_artifact.caption_dir
        image_files = os.listdir(image_dir)

        for img_file in image_files:
            img_name, _ = os.path.splitext(img_file)

            img_path = os.path.join(image_dir, img_file)
            caption_path = os.path.join(caption_dir, img_name + ".txt")

            # Skip if caption doesn't exist
            if not os.path.exists(caption_path):
                continue

            # Read caption
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()

            # Skip empty captions
            if caption == "":
                continue

            pairs.append((img_path, caption_path))

        return pairs
            
    def train(self, dataset):
        logger.info("Starting Model training")
        try:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                    collate_fn=lambda x: self.collate_fn(x, self.processor))
            self.model.train()

            for epoch in range(self.epochs):
                print(f"Epoch {epoch+1}/{self.epochs}")
                pbar = tqdm(dataloader)
                
                for batch in pbar:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(**batch)

                    loss = outputs.loss   # CLIP’s  loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_description(f"Loss: {loss.item():.4f}")
        except Exception as e:
            raise MyException(e,sys)

 
            
    def save_model(self, path):
        logger.info("Saving Model")
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved at {path}")
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_fine_tuner(self):
        logger.info("Starting model fine-tuning pipeline")
        try:

            data = self.load_image_caption_pairs()

            if len(data) == 0:
                raise Exception("No valid image-caption pairs found")

            dataset = ImageCaptionDataset(
                data_list=data,
                processor=self.processor
            )

            self.train(dataset)

            model_save_path = model_fine_tuning_config.trained_model_path
            self.save_model(model_save_path)

            return {
                "model_path": model_save_path,
                "num_samples": len(data),
                "status": "success"
            }

        except Exception as e:
            raise MyException(e, sys)
            
            

            
            
        
