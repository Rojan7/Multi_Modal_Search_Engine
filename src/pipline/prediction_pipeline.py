import sys
from src.entity.config_entity import MultimodalProjConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logger
from PIL import Images


class InputData:
    def __init__(self,Query):
        try:
            self.Query=Query
        except Exception as e:
            raise MyException(e,sys)
        
class MultimodalSearch:
    def __init__(self, multimodal_config : MultimodalProjConfig=MultimodalProjConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = multimodal_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, Query) -> str:
        """
        This is the method of Multimodal search
        Returns: Prediction in string format
        """
        try:
            logger.info("Entered predict method of MultimodalSearch class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(Query)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)