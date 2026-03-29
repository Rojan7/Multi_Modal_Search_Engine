from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.components.Retriver import Retriever
from src.logger import logger
import sys
import os


class Proj1Estimator:
    """
    Handles S3 operations + prediction using Retriever
    """

    def __init__(self, bucket_name, model_path):
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
    



    def is_model_present(self, model_path):
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=model_path
            )
        except MyException as e:
            print(e)
            return False

    def save_model(self, from_folder, remove: bool = False):
        try:
            self.s3.upload_folder(
                folder_path=from_folder,
                bucket_name=self.bucket_name,
                s3_prefix=self.model_path
            )
        except Exception as e:
            raise MyException(e, sys)


    def predict(
        self,
        query,
        top_k=5
    ):
        

        try:
            logger.info("Starting prediction using Retriever")
            faiss_path = os.path.join(self.model_path, "faiss_index","index.bin")
            mapping_path = os.path.join(self.model_path, "mapping.json")

            # Initialize Retriever (loads model + FAISS + mapping)
            retriever = Retriever(
                Faiss_path=faiss_path,
                mapping_path=mapping_path,
                Model_Path=self.model_path
            )

            # Perform search
            results = retriever.predict(query=query, top_k=top_k)

            logger.info("Prediction completed successfully")

            return results

        except Exception as e:
            raise MyException(e, sys)