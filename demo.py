from src.logger import logger
from src.exception import MyException
import sys
import asyncio
from src.pipline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    st=TrainingPipeline()
    st.run_pipline()

# from src.logger import logger
# from src.exception import MyException
# import sys

# try:
#     a = 1+'Z'
# except Exception as e:
#     logger.info(e)
#     raise MyException(e, sys) from e


# from src.logger import logging

# logging.debug("This is a debug message.")
# logging.info("This is an info message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")
# logging.critical("This is a critical message.")