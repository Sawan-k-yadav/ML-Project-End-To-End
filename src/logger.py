# Logger is used for logging any execution or exception or error happening we should be able to log those 
# information in some file so we can be able to track that.

import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # This f string is used to give the name formate of log file which it will create
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)  # What ever log will create it will be stored the path of the logs
os.makedirs(logs_path,exist_ok=True)  # It will keep on appending the file even though there is file in the folder

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # At this point we are going to print this message
)


# if __name__=="__main__":
#     logging.info("Logging has started")