import sys
import asyncio
from src.exception import MyException
from src.logger import logger
from src.components.data_extract import WebCrawler
from src.components.model_loader import CLIPLOADER
from src.components.model_fine_tuning import CLIPFineTuner,ImageCaptionDataset
from src.components.embeddings_generations import EmbeddingGeneration
from src.components.Faiss import FAISSIndexBuilder

from src.entity.config_entity import (data_extract_config,
                                      model_loader_config,
                                      model_fine_tuning_config,
                                      embeddings_generations_config,
                                      Faiss_config)

from src.entity.artifact_entity import( DataExtractorArtifact,
                                        ModelFineTuningArtifact,
                                        ModelLoaderArtifact,
                                        ModelFineTuningArtifact,
                                        EmbeddingGenerationArtifact,
                                        FaissIndexingArtifact)
class TrainingPipeline:
    def __init__(self):
        self.data_extract_config=data_extract_config()
        self.model_loader_config=model_loader_config()
        self.model_fine_tuning_config=model_fine_tuning_config()
        self.embeddings_generations_config=embeddings_generations_config()
        self.Faiss_config=Faiss_config()
        
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
        
    def intiate_embedding_generation(self,
                                     model_training_artifact:ModelFineTuningArtifact,
                                     data_extractor_artifact:DataExtractorArtifact
                                     ):
        try:
            logger.info("Entered initaite_embedding_generation")
            embed_gen=EmbeddingGeneration(
                self.embeddings_generations_config,data_extractor_artifact,
                model_training_artifact
            )
            embeddings_artifact=embed_gen.initiate_embeddings_generations()
            return embeddings_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_index_building(self,
                                embeddings_generations_artifact:EmbeddingGenerationArtifact):
        try:
            logger.info("Entered intiate_index_building method of class TrainPipeline")
            index_builder=FAISSIndexBuilder(
                                            embeddings_generations_artifact)
            faiss_artifact=index_builder.initate_Faiss_index_building()
            logger.info("Index building is successfull")
            return faiss_artifact
        except Exception as e:
            raise MyException(e,sys)
    
    def run_pipline(self):
        data_extract_artifact=self.initiate_crawlling()
        modal_loader_artifact=self.intiate_model_loading(data_extract_artifact)
        trained_model_artifact=self.initiate_model_training(modal_loader_artifact,data_extract_artifact)
        embedding_generation_artifact=self.intiate_embedding_generation(trained_model_artifact,data_extract_artifact)
        faiss_index_builder_artifact=self.initiate_index_building(embedding_generation_artifact)
        
        

        
        

