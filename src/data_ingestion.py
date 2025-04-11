import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_names = self.config["bucket_file_names"]

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info("Data Ingestion Initialized.")

    def download_csv_from_gcp(self):
        try:
            # ✅ Use service account JSON key to authenticate
            credentials_path = r"C:/Users/patil/OneDrive/Desktop/Jayesh_Works/peppy-glyph-454715-c4-f634e4895c0e.json"
            storage_client = storage.Client.from_service_account_json(credentials_path)
            bucket = storage_client.bucket(self.bucket_name)

            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)

                blob = bucket.blob(file_name)
                blob.download_to_filename(file_path)

                if file_name == "animelist.csv":
                    data = pd.read_csv(file_path, nrows=5000000)
                    data.to_csv(file_path, index=False)
                    logger.info("Large file detected. Downloaded only first 5M rows: animelist.csv")
                else:
                    logger.info(f"Downloaded file: {file_name}")

        except Exception as e:
            logger.error("Error while downloading data from GCP.")
            raise CustomException("Failed to download data from GCP.", e)

    def run(self):
        try:
            logger.info("Starting Data Ingestion Process.")
            self.download_csv_from_gcp()
            logger.info("Data Ingestion Completed.")
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
        finally:
            logger.info("Data Ingestion Done.")

if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()
