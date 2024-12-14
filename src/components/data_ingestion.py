import os
import sys 
from src.exceptions import CustomError 
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artefacts","train")
    test_data_path: str = os.path.join("artefacts","test")
    raw_data_path: str = os.path.join("artefacts","data")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion method ")

        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("reading data as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path , index=False , header=True)
            logging.info("Splitting the data ")
            
            train_data ,test_data = train_test_split(df ,test_size=0.2 , random_state=42)
            
            train_data.to_csv(self.ingestion_config.train_data_path , index=False ,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path ,index=False, header = True)

            logging.info("Data Ingestion Completed")
            
            return(
                self.ingestion_config.train_data_path , self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomError(e, sys)

        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
            





