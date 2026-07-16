import yaml
import logging
import logging.config
from settings.settings import settings


def setup_logging() -> None:
    with open(settings.CONFIG_LOG_PATH, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    
    assert logging.getLogger().hasHandlers()
