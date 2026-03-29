import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TimeStamp : str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name : str = PIPELINE_NAME
    artifact_dir :str = os.path.join(ARTIFACT_DIR,TimeStamp)
    timestamp :str = TimeStamp
    
training_pipeline_config: TrainingPipelineConfig=TrainingPipelineConfig()
    

@dataclass 
class data_extract_config:
    data_extract_output_dir:str=os.path.join(training_pipeline_config.artifact_dir,EXTRACTED_DIR_NAME)
    
@dataclass
class model_loader_config:
    model_dir:str=os.path.join(training_pipeline_config.artifact_dir,LOAD_MODEL_DIR,"CLIP_MODEL_UNTRAINED")

    
@dataclass
class model_fine_tuning_config:
    trained_model_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        train_model_dir,
        "Fine_Tuned_Model"
    )
    
    model_save_dir: str = os.path.join(
        training_pipeline_config.artifact_dir
    )

    
@dataclass
class embeddings_generations_config:
    embeddings_path:str = os.path.join(training_pipeline_config.artifact_dir,embeddings_dir,"Embeddings")
    Mapping_path:str = os.path.join(training_pipeline_config.artifact_dir,embeddings_dir,"Mappnig")


@dataclass
class Faiss_config:
    index_path:str = os.path.join(training_pipeline_config.artifact_dir,embeddings_dir,"index")
    
# @dataclass
# class Retriver_Config:
    
@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME
    
    # faiss_index_path: str
    # mapping_path: str
    
    
@dataclass
class MultimodalProj:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME
    
