import os 
import sys 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from dataclasses import dataclass 
import pandas as pd
import numpy as np
from src.logger import logging 
from src.exceptions import CustomError 
from src.utils import save_object

@dataclass
class TansformationConfig:
    preprocessor_obj_file_path = os.path.join('artefacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = TansformationConfig()

    def get_transformer_obj(self):
        ''' This Function Is responsible for data transformation '''
        try:
            numerical_columns = ["writing_score" ,"reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median" )),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder()),
                    ("scaler",StandardScaler( with_mean=False))
                ]
            )
            logging.info("numerical Column encoding Completed")
            logging.info("categorical Columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline , numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomError(e,sys)

    def initiate_data_transformation(self, train_data_path , test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info("Training and Test Data reading completedd")
            logging.info("Obtaining preprocessing Object ")

            preprocessing_obj = self.get_transformer_obj()
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score" ,"reading_score"] 
            input_feature_train_df = train_data.drop(columns = [target_column_name] , axis = 1  )
            target_feature_train_df  = train_data[target_column_name]

            input_feature_test_df = test_data.drop(columns = [target_column_name],axis=1)
            target_feature_test_df = test_data[target_column_name]

            logging.info(f"appling preprocessing to training and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            

            save_object(file_path =self.data_transformation_config.preprocessor_obj_file_path, obj = preprocessing_obj)


            logging.info("saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomError(e,sys)
        


