import os
import sys
import logging


LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


def setup_model_logger(model_name: str) -> logging.Logger:
    logger = logging.getLogger(f'model.{model_name}')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(LOG_DIR, f'{model_name}.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def setup_optuna_logger(logger: logging.Logger) -> None:
    optuna_logger = logging.getLogger('optuna')
    optuna_logger.handlers = []
    for handler in logger.handlers:
        optuna_logger.addHandler(handler)
    optuna_logger.setLevel(logging.INFO)
