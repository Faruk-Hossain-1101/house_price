import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exeception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprosser_obj_file_path = os.path.join('artifcats', 'preprosser.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info("Data Transfromation Initiated")
            
            ## Define which columns are categorical and numecarical
            categotical_col = ['cut', 'color', 'clarity']
            numerical_col = ['carat', 'depth', 'table', 'x', 'y', 'z']

            ## Define the categorical categoris
            cut_cat = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_cat = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_cat = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info("Pipeline Inisited")

            ## Numecrial pipeline
            num_pupeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy= 'median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('inputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_cat, color_cat, clarity_cat])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pupeline', num_pupeline, numerical_col),
                ('cat_pipeline', cat_pipeline, categotical_col)
            ])

            logging.info("Pipeline completed")

            return preprocessor


        except Exception as e:
            logging.info("Error in Data Transfronation")
            raise CustomException(e, sys)
        
    def initatite_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            logging.info("Read Train Test Data")
            logging.info(f'Train Data Head: \n{train_df.head().to_string()}' )
            logging.info(f'Test Data Head: \n{test_df.head().to_string()}' )
            
            logging.info('Obtaining preprossing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_col_name = "price"
            drop_col = [target_col_name, 'id']

            input_feature_train_df = train_df.drop(columns=drop_col, axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=drop_col, axis=1)
            target_feature_test_df = test_df[target_col_name]
            
            logging.info("Appling processing object on training and test dataset")

            ## Tramsffroming using preprocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(input_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(input_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprosser_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor file save')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprosser_obj_file_path
            )
        except Exception as e:
            logging.info("Execption occoured in initatite data transfromation")
            raise CustomException(e, sys)