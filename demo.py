from src.logger import logger
from src.exception import MyException
import sys
import asyncio
from src.components.data_ingestion.crawler import WebCrawler

if __name__ == "__main__":
    seed_urls = ["https://en.wikipedia.org/wiki/War"]
    crawler = WebCrawler(seed_urls)
    text_data, image_data = asyncio.run(crawler.run())
    print(f"Text docs: {len(text_data)}, Images: {len(image_data)}")

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