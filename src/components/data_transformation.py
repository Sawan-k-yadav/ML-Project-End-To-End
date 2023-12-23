# This file will have all the data transformation technique and its logic

import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer  # It is used mainly for creating pipeline of onehot encoding or stabd scaling
from sklearn.impute import SimpleImputer  # For checking missign value
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig():   # It will have all the inputs which requires for data trasnformation
    preprocessor_obj_file_path=os.path.join('artifacts',"processor.pkl")  # If we are creating any model and saving it in any pikle file this is the varaible to use it for saving


class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):   # This will create pikle file which will be used to trasnform categorical feature into numerical or standard scaler, transformation etc
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),   # This handling missing value and outlier with medium strategy
                    ("scaler",StandardScaler(with_mean=False))   # By setting with_mean=False, the StandardScaler does not subtract the mean from each feature, and instead scales the features based on their variance. This can help preserve the positive values of the features and avoid issues.
                ]
            )

            cat_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),  # Here also used for handling missing value and if needed replacing with mode value
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical column : {numerical_columns}")
            logging.info(f"Categorical column : {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):  # it will be used as data transformation by taking training and testing data from data_ingestion,py file

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining Preprocessor object")

            preprocessing_obj=self.get_data_transformation_object()

            target_column_name="math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            
            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )


            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



                         
        except Exception as e:
            raise CustomException(e, sys)
