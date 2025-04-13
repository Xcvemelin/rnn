"""
rnn-text-generator package initialization
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from configs.logging import configure_logger
logger = configure_logger()

logger.debug(f"Starting package initialization - {__name__}")

__version__ = "0.1.0"
__author__ = "xcvemelin"
__license__ = "MIT"
__description__ = "Package for text generation using recurrent neural networks"

try:
    from .data.dataset_generator import DatasetGenerator
    from .utils.text_generator import TextGenerator
    from .models.model_factory import ModelFactory
    from .configs.model_config import ModelConfig
    
    logger.debug("Main components imported successfully")
except ImportError as e:
    logger.critical(f"Import failed: {e}")
    raise

__all__ = [
    'DatasetGenerator',
    'TextGenerator',
    'ModelFactory',
    'logger',
    'ModelConfig'
]

try:
    import tensorflow as tf
    import numpy as np
    logger.debug("Required dependencies are available")
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    raise

DEFAULT_MODEL = "rnn"
DEFAULT_TEMPERATURE = 0.7

logger.info(f"Package {__name__} v{__version__} initialized successfully")