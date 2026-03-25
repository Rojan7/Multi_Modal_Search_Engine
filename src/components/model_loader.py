import torch
from transformers import CLIPModel, CLIPProcessor
from src.entity.artifact_entity import ModelLoaderArtifact,DataExtractorArtifact
from src.entity.config_entity import model_loader_config,data_extract_config
from src.constants import MODEL_URL,USE_HALF
from src.logger import logger
from src.exception import MyException
import sys



class CLIPLOADER:
    def __init__(self,model_loader_config:model_loader_config,
                 data_extractor_artifact:DataExtractorArtifact):
        self.model_name = MODEL_URL
        self.device= torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model= None
        #self.load_model()
        self.use_half=USE_HALF
        self.model_loader_config=model_loader_config
        
    # Model Loading
    def load_model(self):
        try:
            logger.info("Loading Clip model")
            self.model=CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor=CLIPProcessor.from_pretrained(self.model_name)
            if self.use_half and self.device == "cuda":
                self.model = self.model.half()
            self.model.eval()
            self.model.save_pretrained(model_loader_config.model_dir)
            self.processor.save_pretrained(model_loader_config.pro_dir)
            logger.info("Model loaded and saved sucessfuly")
        except Exception as e:
            raise MyException(e,sys)
        
    #Text encoding
    def encode_text(self,texts,normalize=True):
        try:
            logger.info("Encoding text")
            inputs=self.processor(
                text=texts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            if normalize:
                text_features = torch.nn.functional.normalize(text_features,p=2,dim=-1)
            logger.info("text encoded sucessfully")
            return text_features
        except Exception as e:
            raise MyException(e,sys)
    #Image encoding
    def encode_image(self,images,normalize=True):
        try:
            logger.info("Encoding images")
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            if normalize:
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            logger.info("Image encoded successfully")
            return image_features
        except Exception as e:
            raise MyException(e,sys)

    def compute_similarity(self, image_embeds, text_embeds):
        try:
            logger.info("Computing Cosine similarity")
            return image_embeds @ text_embeds.T
        except Exception as e:
            raise MyException(e,sys)
    
    def initiate_model_loading(self)->ModelLoaderArtifact:
        logger.info("Entered initiate model loading")
        try:
            self.load_model()
            model_loader_artifact=ModelLoaderArtifact(
                LoadedModelPath=model_loader_config.model_dir,
                LoadedProcPath=model_loader_config.pro_dir
            )
            return model_loader_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    
    