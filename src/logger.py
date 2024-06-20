import logging
import logging.config

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

# get root logger
logger = logging.getLogger()
