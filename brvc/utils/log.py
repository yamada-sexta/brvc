import logging
logger = logging.getLogger(__name__)

def setup_logger():
    from pythonjsonlogger.json import JsonFormatter
    handler = logging.StreamHandler()
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)