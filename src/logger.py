import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Directory for logs
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file path with dynamic date in the name
LOG_FILE = os.path.join(LOGS_DIR, f"logs_{datetime.now().strftime('%Y-%m-%d')}.log")

# Set up logging configuration
def setup_logger(name):
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # Log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # Rotating File Handler (5MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Logger level
    logger.setLevel(logging.INFO)

    return logger

# Expose function
def get_logger(name='AnimeRecommenderLogger'):
    return setup_logger(name)
