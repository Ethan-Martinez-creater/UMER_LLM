"""Training logger utility."""
import os
import logging
from datetime import datetime


def init_train_logger(output_dir: str) -> logging.Logger:
    """Initialize dual-output logger (console + file)."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_file)

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("TRAINING LOG START")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 60)

    return logger
