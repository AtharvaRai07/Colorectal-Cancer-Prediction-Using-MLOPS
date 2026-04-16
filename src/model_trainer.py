import sys
import os
from contextlib import nullcontext
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from config.paths_config import *
from config.model_params import *
from src.logger import logging
from src.exception import CustomException

class ModelTrainer:
    def __init__(self, input_file_path: str, model_file_path: str, scaler_file_path: str, model_test_result: str):
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.scaler_file_path = scaler_file_path
        self.model_test_result = model_test_result

        self.gradient_boosting_params = GRADIENT_BOOSTING_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        self.model = None

        self.data = None
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.scaler_file_path), exist_ok=True)

    def load_data(self):
        try:
            logging.info(f"Loading data from {self.input_file_path}")

            self.data = pd.read_csv(self.input_file_path)

            logging.info("Data loaded successfully")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomException(e, sys)

    def split_data(self):
        try:
            logging.info("Splitting data into train and test sets")

            self.X = self.data.drop('Survival_Prediction', axis=1)
            self.y = self.data['Survival_Prediction']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            logging.info("Data split successfully")
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise CustomException(e, sys)

    def scale_data(self):
        try:
            logging.info("Scaling data using StandardScaler")

            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

            joblib.dump(scaler, self.scaler_file_path)

            logging.info(f"Data scaled and scaler saved at {self.scaler_file_path}")

        except Exception as e:
            logging.error(f"Error scaling data: {e}")
            raise CustomException(e, sys)

    def train_model(self):
        try:
            logging.info("Initializing Model")

            gbc = GradientBoostingClassifier(random_state= self.random_search_params['random_state'])

            random_search = RandomizedSearchCV(
                estimator=gbc,
                param_distributions=self.gradient_boosting_params,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                scoring=self.random_search_params['scoring']
            )

            logging.info("Starting model training with RandomizedSearchCV")

            random_search.fit(self.X_train, self.y_train)

            logging.info("Model training completed")
            logging.info(f"Best parameters found: {random_search.best_params_}")
            logging.info(f"Best score achieved: {random_search.best_score_}")

            self.model = random_search.best_estimator_

            joblib.dump(self.model, self.model_file_path)

            logging.info(f"Model trained and saved at {self.model_file_path}")

        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise CustomException(e, sys)

    def evaluate_model(self):
        try:
            logging.info("Evaluating model")

            best_model = self.model
            y_pred = best_model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, best_model.predict_proba(self.X_test)[:, 1])

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            with open(self.model_test_result, 'w') as f:
                f.write(f"Model Evaluation Results:\n")
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"ROC AUC Score: {roc_auc}\n")

            logging.info(f"Model evaluation completed and results saved at {self.model_test_result}")

        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise CustomException(e, sys)

    def run(self):
        try:
            logging.info("Starting model training process...")

            self.load_data()
            self.split_data()
            self.scale_data()
            self.train_model()
            self.evaluate_model()

            logging.info("Model training process completed successfully")

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    run_context = mlflow.start_run() if MLFLOW_AVAILABLE else nullcontext()
    with run_context:
        model_trainer = ModelTrainer(input_file_path=PREPROCESSED_FILE_PATH, model_file_path=MODEL_FILE_PATH, scaler_file_path=SCALER_FILE_PATH, model_test_result=MODEL_TEST_RESULT)
        model_trainer.run()
