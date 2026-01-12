import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with a standard format and both file/stream handlers.

    Args:
        name: Name of the logger.
        log_level: logging level (e.g., logging.INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if the logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Stream Handler (to console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler (to logs/ folder)
        log_dir = Path(__file__).parent.parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'pipeline.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
