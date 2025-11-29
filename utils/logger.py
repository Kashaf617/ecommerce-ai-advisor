"""
Logging utility for the E-Commerce Business Automation Platform
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from config import LOGGING_CONFIG

def setup_logger(name: str, log_file: Path = None, level: str = None) -> logging.Logger:
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name (usually __name__ of the module)
        log_file: Path to log file (optional, uses config default if not provided)
        level: Logging level (optional, uses config default if not provided)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    log_level = getattr(logging, level or LOGGING_CONFIG['level'])
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG['format'])
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = LOGGING_CONFIG['log_file']
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    return setup_logger(name)
