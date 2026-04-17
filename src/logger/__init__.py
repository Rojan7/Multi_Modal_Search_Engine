import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

#constant for log configurations
log_dir="logs"
log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
max_log_size=5*1024*1024
back_up_count=3

#Construct log file path
log_dir_path=os.path.join(from_root(),log_dir)
os.makedirs(log_dir_path,exist_ok=True)
log_file_path=os.path.join(log_dir_path,log_file)

def configure_logger():
    logger=logging.getLogger()
    logger.setLevel("DEBUG")
    
    formater=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    #File handeler with rotations
    file_handeller=RotatingFileHandler(log_file_path,maxBytes=max_log_size,backupCount=back_up_count)
    file_handeller.setFormatter(formater)
    file_handeller.setLevel("DEBUG")
    
    console_handeller=logging.StreamHandler()
    console_handeller.setLevel("DEBUG")
    console_handeller.setFormatter(formater)
    
    logger.addHandler(console_handeller)
    logger.addHandler(file_handeller)
    return logger

logger=configure_logger()