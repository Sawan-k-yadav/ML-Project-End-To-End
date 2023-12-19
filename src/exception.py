import sys
from src.logger import logging


# This function will be used to log the custom error if we are getting any inside the project
def error_message_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()   # error_detail.exc_info() will give error detail and it will be store exc_tb variable to get error file name, error message, error line
    file_name=exc_tb.tb_frame.f_code.co_filename # for getting file name
    error_message="Error occured in Python script name [{0}] line number [{1}] error message [{2}]".format(  
     file_name,exc_tb.tb_lineno,str(error)) # These [{0}], [{1}], [{2}] are placeholder for message to display seperated way

    return error_message

    


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)   # Inheriting error from Exception class
        self.error_message=error_message_detail(error_message, error_detail=error_detail)  
    

    def __str__(self):          # For printing the error message
        return self.error_message
    

# if __name__=="__main__":

#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by zero error")
#         CustomException(e, sys)
    