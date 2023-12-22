# This file keeps all data which are raw or training or testing etc.

import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass                       # This library is used to create class variable


# Below class is used to initialize input data 
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")  #artifacts used for saving outputs in train.csv file
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()  # In this variable all the data will be initialize   || I was getting error when I first ran data_ingestion.py file because I had missed to add DataingestionConfig and add Dataingestion() only


    def initiate_data_ingestion(self):    # This function will be used to read the data if we are reading the data from any database or any data source
        logging.info("ENtered data ingestion method or components")
        try:
            df=pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Read the dataset in dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)  # It will check if the folder present then we will kep the folder so that we dont need to recreate it

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)  # data which we read as csv, saving it in raw data

            logging.info('Train test split initialted')

            train_set, test_set =train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)  # Storing train and test data in separte variables for data transformation

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)   # It will raise my custom exception



if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()


