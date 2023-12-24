# It is used to predict the model with some unseen data using one small web application

import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):  # It will be used as predicting model
           try:
                print(features.columns)
                model_path='artifacts\model.pkl'
                preprocessor_path='artifacts\processor.pkl' # This variable is use for EDA, preprocessing etc.
                model=load_object(file_path=model_path)  # it is used for loading the pickle file which we are using
                preprocessor=load_object(file_path=preprocessor_path)
                data_scaled=preprocessor.transform(features)  # We are scaling the data
                preds=model.predict(data_scaled)  # Predictign with scaled data
                return preds 

           except Exception as e:
               raise CustomException(e, sys)




class CustomData:
    def __init__(
            self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):   # It will create all data as dataframe as our model need dataframe type data
        try:
            custom_data_input_dict={   # This will create variable and then it will assign all the custructor value which we got from our web apps
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)