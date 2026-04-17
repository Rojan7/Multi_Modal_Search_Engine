import sys
from src.logger import logger

def error_message_details(error:Exception,error_details:sys)->str:
    '''
    Docstring for error_message_details
    
    :param error: Exception raised 
    :type error: Exception
    :param error_details: Description
    :type error_details: sys
    :return: Description
    :rtype: str
    '''
    
    #Extract traceback details 
    
    _,_,exc_tb= error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    line_number=exc_tb.tb_lineno
    error_message=f"Error occured in python script [{file_name}] at line number [{line_number}]"
    logger.error(error_message)
    return error_message
    
    
class MyException(Exception):
    # custom exception class inheriting from pythons standard Exception Class
    def __init__(self, error_message:str,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail)
        
    
    def __str__(self)->str:
        return self.error_message