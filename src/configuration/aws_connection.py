import boto3
import os
from src.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY,REGION_NAME

class s3Client:
    s3_client=None
    s3_resource=None
    
    def __init__(self,region_name=REGION_NAME):
        
        _access_key_id=os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY,)
        _secret_access_key=os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY,)
            
        if _access_key_id==None or _secret_access_key==None:
            raise Exception("Environment variable for aws acess key or scret key are not set.")
        
        s3Client.s3_resource=boto3.resource(
            's3',
            aws_access_key_id=_access_key_id,
            aws_secret_access_key=_secret_access_key,
            region_name=region_name
        )
        
        
        s3Client.s3_client=boto3.client(
            's3',
            aws_access_key_id=_access_key_id,
            aws_secret_access_key=_secret_access_key,
            region_name=region_name       
        )
        
        self.s3_resource=s3Client.s3_resource
        self.s3_client=s3Client.s3_client