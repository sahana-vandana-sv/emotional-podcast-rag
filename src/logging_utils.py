import logging
from pathlib import Path
from src.config import LOG_FILE


def get_logger(
    name: str,
    log_file: str = None,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO
    )-> logging.Logger:

     # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter

    # Avoid duplicate handlers if function is called multiple times
    if logger.handlers:
        logger.handlers.clear()

    # Determine log file path
    if log_file is None:
        log_file = f"{name}.log"
    
    log_path = LOG_FILE

    # ──────────────────────────────────────────────────────
    # File handler (writes everything to disk)
    # ──────────────────────────────────────────────────────
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # ──────────────────────────────────────────────────────
    # Console handler (writes to terminal/notebook)
    # ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(
        "%(levelname)-8s | %(message)s"
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_quiet_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Get a logger that only writes to file (no console output).
    
    Useful for verbose operations where you want logs for debugging
    but don't want to spam the console.
    """
    return get_logger(
        name=name,
        log_file=log_file,
        file_level=logging.DEBUG,
        console_level=logging.CRITICAL,  # Effectively silent
    )


def get_verbose_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Get a logger that shows DEBUG messages in console.
    
    Useful for development/debugging.
    """
    return get_logger(
        name=name,
        log_file=log_file,
        file_level=logging.DEBUG,
        console_level=logging.DEBUG,
    )

