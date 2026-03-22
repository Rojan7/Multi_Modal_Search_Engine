from src.logger import logger
from src.exception import MyException
import sys
import asyncio
from src.components.data_extract import WebCrawler

if __name__ == "__main__":
    seed_urls = ["https://www.alibaba.com/?from=ugclickserver&isSpider=true"]
    
    crawler = WebCrawler(seed_urls)
    results = asyncio.run(crawler.run())
  

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