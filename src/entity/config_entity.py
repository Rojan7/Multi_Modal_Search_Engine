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
    data_extract_output_dir=os.path.join(training_pipeline_config.artifact_dir,EXTRACTED_DIR_NAME)
    
    
