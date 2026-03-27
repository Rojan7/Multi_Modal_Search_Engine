from dataclasses import dataclass, field

@dataclass 
class DataExtractorArtifact:
    image_dir : str
    caption_dir:str
    Img_to_url : dict[tuple, str] = field(default_factory=dict)
    
@dataclass
class ModelLoaderArtifact:
    LoadedModelPath : str 
    
@dataclass 
class ModelFineTuningArtifact:
    Model_Path:str
    Status:str
    Num_Samples:str
    
@dataclass
class EmbeddingGenerationArtifact:
    Embeddings_path:str
    status:str
    
@dataclass
class FaissIndexingArtifact:
    Faiss_path:str
    status:str