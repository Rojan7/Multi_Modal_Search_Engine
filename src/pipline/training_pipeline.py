import sys
import asyncio
from src.exception import MyException
from src.logger import logger
from src.components.data_extract import WebCrawler
from src.components.model_loader import CLIPLOADER
from src.components.model_fine_tuning import CLIPFineTuner,ImageCaptionDataset

from src.entity.config_entity import (data_extract_config,
                                      model_loader_config,
                                      model_fine_tuning_config)

from src.entity.artifact_entity import( DataExtractorArtifact,
                                        ModelFineTuningArtifact,
                                        ModelLoaderArtifact)
class TrainingPipeline:
    def __init__(self):
        self.data_extract_config=data_extract_config()
        self.model_loader_config=model_loader_config()
        self.model_fine_tuning_config=model_fine_tuning_config()
        
        # self.data_extract_artifact=DataExtractorArtifact()
        # self.model_fine_tuning_artifact=ModelFineTuningArtifact()
        # self.model_loader_artifact=ModelLoaderArtifact()
        
    def initiate_crawlling(self)-> DataExtractorArtifact:
        logger.info("Entered intiate_crawlling method of class TrainingPipeline")
        try:
            data_crawller=WebCrawler(self.data_extract_config)
            data_crawller_artifact=asyncio.run(data_crawller.run())
            return data_crawller_artifact
        except Exception as e:
            raise MyException(e,sys)
    
    def intiate_model_loading(self,data_extractor_artifact:DataExtractorArtifact)-> ModelLoaderArtifact:
        logger.info("Entered initiate Model loading method of class TrainingPipeline")
        try:
            model_loader=CLIPLOADER(self.model_loader_config,
                                    data_extractor_artifact)
            model_loader_artifact=model_loader.initiate_model_loading()
            logger.info("initiate_model_loading method of class TrainPipeline is sucessfull")
            return model_loader_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_training(self,
                                model_loader_artifact:ModelLoaderArtifact,
                                data_extractor_artifact:DataExtractorArtifact):
        logger.info("Entered initate_model_training of class TrainingPipeline")
        try:
            training_model=CLIPFineTuner(self.model_fine_tuning_config,model_loader_artifact,data_extractor_artifact)
            model_train_artifact=training_model.initiate_model_fine_tuner()
            return model_train_artifact
        except Exception as e:
            raise MyException(e,sys)
    
    def run_pipline(self):
        data_extract_artifact=self.initiate_crawlling()
        modal_loader_artifact=self.intiate_model_loading(data_extract_artifact)
        self.initiate_model_training(modal_loader_artifact,data_extract_artifact)
        

        
        

