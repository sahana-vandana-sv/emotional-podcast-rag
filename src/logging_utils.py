# src/logging_utils.py
import logging
import os
from pathlib import Path


def get_logger(
    name: str,
    log_file: str = None,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO
) -> logging.Logger:
    """
    Get a configured logger with file and console handlers.
    
    Automatically creates log directory if it doesn't exist.
    
    Parameters
    ----------
    name : str
        Logger name
    log_file : str, optional
        Log file name (will be created in logs/ directory)
    file_level : int
        Logging level for file handler
    console_level : int
        Logging level for console handler
    
    Returns
    -------
    logging.Logger
    """
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # ─────────────────────────────────────────────────────
    # Console handler (always add)
    # ─────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter('%(levelname)-8s | %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # ─────────────────────────────────────────────────────
    # File handler (only if log_file specified)
    # ─────────────────────────────────────────────────────
    if log_file:
        try:
            # Determine log path
            if os.path.isabs(log_file):
                log_path = Path(log_file)
            else:
                # Relative path - put in logs/ directory at project root
                project_root = Path(__file__).parent.parent
                log_dir = project_root / "logs"
                log_path = log_dir / log_file
            
            # Create directory if it doesn't exist
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(file_level)
            file_format = logging.Formatter(
                '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # If file logging fails, just log to console
            logger.warning(f"Could not create file handler: {e}")
    
    return logger


def get_quiet_logger(name: str, log_file: str = None) -> logging.Logger:
    """Get logger that only logs to file (no console output)."""
    return get_logger(name, log_file, file_level=logging.DEBUG, console_level=logging.CRITICAL)


def get_verbose_logger(name: str, log_file: str = None) -> logging.Logger:
    """Get logger with DEBUG level to console."""
    return get_logger(name, log_file, file_level=logging.DEBUG, console_level=logging.DEBUG)