import sys
from datetime import datetime
from configs.paths import LOGS_DIR

def configure_logger():
    ''' Configuration of logger for all project '''
    from loguru import logger
    logger.remove()
    current_time = datetime.now().strftime("%Y-%m-%d")
    
    # Log output to the console
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Logs output to the file
    logger.add(
       LOGS_DIR / f"rnn_log_{current_time}.log",
        rotation="50 MB",
        retention="7 days",
        compression="zip",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    return logger

logger = configure_logger()