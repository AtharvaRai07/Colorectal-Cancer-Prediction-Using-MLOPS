import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.common_functions import read_yaml
from config.paths_config import *
from src.logger import logging
from src.exception import CustomException

class DataProcessing:
    def __init__(self, input_file_path: str, output_file_path: str, label_encoder_file_path: str, config_path: str):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.label_encoder_file_path = label_encoder_file_path
        self.config_path = read_yaml(config_path)
        self.k = self.config_path['data_preprocessing']['k']

        self.label_encoder = {}
        self.scaler = StandardScaler()
        self.df = None
        self.target_column = None

        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.label_encoder_file_path), exist_ok=True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_file_path)
            logging.info(f"Data loaded successfully from {self.input_file_path}")

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomException(e, sys)

    def preprocess_data(self):
        try:
            self.df = self.df.drop('Patient_ID', axis=1)

            logging.info("Created X and y from dataset")
            cat_cols = self.df.select_dtypes(include=['object']).columns

            for col in cat_cols:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoder[col] = le

            logging.info("Encoded categorical features")

        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise CustomException(e, sys)

    def feature_selection(self):
        try:
            logging.info("Selecting features using f-test...")
            selector = SelectKBest(f_classif, k="all")

            X = self.df.drop('Survival_Prediction', axis=1)

            selector.fit(X, self.df['Survival_Prediction'])

            feature_scores = pd.DataFrame({
                    'Feature': X.columns,
                    'F_Scores': selector.scores_
            }).sort_values(by='F_Scores', ascending=False).reset_index(drop=True)

            self.selected_features = feature_scores.head(self.k)['Feature'].tolist()
            logging.info(f"Selected top {self.k} features: {self.selected_features}")

            self.target_column = self.df['Survival_Prediction'].copy()
            self.df = self.df[self.selected_features].copy()

        except Exception as e:
            logging.error(f"Error in feature selection: {e}")
            raise CustomException(e, sys)

    def save_artifacts(self):
        try:
            logging.info("Saving preprocessed data...")
            preprocessed_df = self.df.copy()
            preprocessed_df['Survival_Prediction'] = self.target_column

            # Verify data integrity before saving
            if len(preprocessed_df) == 0:
                raise ValueError("No data to save after preprocessing")

            preprocessed_df.to_csv(self.output_file_path, index=False)
            logging.info(f"Preprocessed data saved to {self.output_file_path}")

            logging.info("Saving label encoders...")
            joblib.dump(self.label_encoder, self.label_encoder_file_path)
            logging.info(f"Label encoders saved to {self.label_encoder_file_path}")

        except Exception as e:
            logging.error(f"Error in saving artifacts: {e}")
            raise CustomException(e, sys)

    def run(self):
        try:
            self.load_data()
            self.preprocess_data()
            self.feature_selection()
            self.save_artifacts()

        except Exception as e:
            logging.error(f"Error in data processing: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    logging.info("Starting data processing...")

    data_processor = DataProcessing(input_file_path=RAW_FILE_PATH, output_file_path=PREPROCESSED_FILE_PATH, label_encoder_file_path=LABEL_ENCODER_FILE_PATH, config_path=CONFIG_PATH)
    data_processor.run()

    logging.info("Data processing completed successfully.")


