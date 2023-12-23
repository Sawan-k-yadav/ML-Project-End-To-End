# Here we will train different model to check thier accuracy other performace checking parameter

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()  # this object wlll have path of trainer_model_file_path

    # def initiate_model_trainer(self, train_array, test_array, Preprocessor_path): -- this Preprocessor_path is removed after creating model trainer file complete and importing it in data ingestion file
    def initiate_model_trainer(self, train_array, test_array):   # it will initiate trainign model by taking data transformation variable
        try:
            logging.info("Split train and test input data ")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],    # Mean just take out the last element and store rest all elemenst fron train_array to X_train
                train_array[:,-1],  # means taking last element of train_arraya nd storing in y_train
                test_array[:,:-1],    # Mean just take out the last element and store rest all elemenst fron test_array to X_test
                test_array[:,-1]     # means taking last element of test_array and storing in y_test
            )

            models = {                                          # Creating dictionary of models to apply the traing through loop in every models
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, 
                                              models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model Found")
            
            logging.info(f"Best model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test, predicted)  
            print(r2_square)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


