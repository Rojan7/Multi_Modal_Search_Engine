import sys
import os

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logger
from src.entity.artifact_entity import ModelPusherArtifact, ModelFineTuningArtifact,FaissIndexingArtifact,EmbeddingGenerationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.s3 = SimpleStorageService()
        self.model_pusher_config = model_pusher_config

        self.proj1_estimator = Proj1Estimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(
        self,
        model_fine_tuning_artifact: ModelFineTuningArtifact,
        faiss_artifact:FaissIndexingArtifact,
        embedding_artifact: EmbeddingGenerationArtifact
    ) -> ModelPusherArtifact:

        logger.info("Entered ModelPusher")

        try:
            

            logger.info("Uploading model to S3...")
            self.proj1_estimator.save_model(
                from_folder=model_fine_tuning_artifact.Model_Path
            )


            logger.info("Uploading FAISS index...")

            self.s3.upload_folder(
                folder_path=faiss_artifact.Faiss_path,
                bucket_name=self.model_pusher_config.bucket_name,
                s3_prefix=os.path.join(self.model_pusher_config.s3_model_key_path, "faiss_index")
            )


            logger.info("Uploading mapping file...")

            self.s3.upload_file(
                from_file=embedding_artifact.mapping_path,
                to_file=os.path.join(
                    self.model_pusher_config.s3_model_key_path,
                    "mapping.json"
                ),
                bucket_name=self.model_pusher_config.bucket_name
                
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logger.info(f"Model pushed successfully: {model_pusher_artifact}")

            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e