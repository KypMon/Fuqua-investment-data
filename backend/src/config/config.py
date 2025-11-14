#import os 
import socket
import sys
from dotenv import dotenv_values
from src.logging.app_logger import AppLogger

class Config(object):
    config = None

    @staticmethod
    def set_up_config(config_file_path:str) -> dict:
        logger = AppLogger.get_logger()
        
        try:
            host_name = socket.gethostname()
            host_ip = socket.gethostbyname(host_name)
        except Exception as e:
            logger.error(str(e))
            sys.exit(99)

        config = dotenv_values(config_file_path)

        logger.info("CONFIGURATION BEGIN : ******************************************************")
        logger.info("These are the configuration values")
        for key,value in config.items():
            logger.info("CONFIGURATION: " + key + " -> " + value)
        logger.info("CONFIGURATION END   : ******************************************************")
        logger.info("ENVIRONMENT: The machine host name and IP is: " + host_ip + " " + host_name)
        
        Config.config = config
        return Config.config

    @staticmethod
    def get_property(key:str) -> str:
        return Config.get_config().get(key)
    
    @staticmethod
    def get_config():
        return Config.config
    
#Config.get_config()