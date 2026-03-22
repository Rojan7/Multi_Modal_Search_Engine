from dataclasses import dataclass, field

@dataclass 
class DataExtractorArtifact:
    Img_to_url : dict[str , tuple] = field(default_factory=dict)
    
    
    
    
    
    

