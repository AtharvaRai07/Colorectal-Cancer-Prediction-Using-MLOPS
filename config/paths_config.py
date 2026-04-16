from pathlib import Path
import os

############################ DATA INGESTION PATHS ############################
RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "data.csv")

CONFIG_PATH = "config/config.yaml"

############################ DATA PROCESSING PATHS ###########################
PREPROCESSED_DIR = "artifacts/preprocessed"
PREPROCESSED_FILE_PATH = os.path.join(PREPROCESSED_DIR, "preprocessed_data.csv")

LABEL_ENCODER_DIR = os.path.join(PREPROCESSED_DIR, "label_encoder")
LABEL_ENCODER_FILE_PATH = os.path.join(LABEL_ENCODER_DIR, "label_encoder.pkl")

############################# MODEL TRAINING PATHS ############################
MODEL_DIR = "artifacts/model"

SCALER_DIR = os.path.join(MODEL_DIR, "scaler")
SCALER_FILE_PATH = os.path.join(SCALER_DIR, "X_train_scaler.pkl")

MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.pkl")
MODEL_TEST_RESULT = os.path.join(MODEL_DIR, "model_evaluation.txt")
