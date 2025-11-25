import os
from datetime import datetime, UTC
import pandas as pd
from src.config.config import Config
from src.logging.app_logger import AppLogger
from src.middleware.fw_user import FwUser

class FileService(object):

    def __init__(self):
        self.logger = AppLogger.get_logger()

        config = Config.get_config()

        data_dir = config.get("data.directory")

        # stocks_mf_ETF_data_final.csv
        self.etf_file = os.path.join(data_dir, config.get("stocks.etf"))

        # F-F_Research_Data_Factors.csv
        self.ff_file = os.path.join(data_dir, config.get("f.f.research.data.factors"))

        # F-F_Research_Data_5_Factors_2x3.csv
        self.ff5_file = os.path.join(data_dir, config.get("f.f.research.data.5.factors.2by3"))

        # F-F_Momentum_Factor.csv
        self.mom_file = os.path.join(data_dir, config.get("f.f.momentum.factor"))

        self.STATIC_DIR = config.get("static.dir")

    def get_etf_file_path(self) -> str:
        return self.etf_file
    
    def get_ff_file_path(self) -> str:
        return self.ff_file
    
    def get_ff5_file_path(self) -> str:
        return self.ff5_file
    
    def get_mom_file_path(self) -> str:
        return self.mom_file
    
    def get_STATIC_DIR(self) -> str:
        return self.STATIC_DIR
    
    def save_dataframe(self, user, df: pd.DataFrame, prefix: str) -> str:
        filename = self.timestamped_filename(prefix)
        filename = self.user_timestamped_filename(filename, user) # embed userId
        path = os.path.join(self.STATIC_DIR, filename)
        self.logger.info("Saving " + path + " to dataframe")
        df.to_csv(path, index=False)
        return filename
    
    def timestamped_filename(self, prefix: str) -> str:
        # ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")  # utcnow method is deprecated in python 3.13
        ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f") 
        return f"{prefix}_{ts}.csv"
    
    def user_timestamped_filename(self, fn:str, user:FwUser) ->str:
        parts = fn.split(".")
        user_filename = parts[0] + "_" + user.get_userid() + "." + parts[1]
        return user_filename
    
    def make_static_dir(self):
        self.logger.info("Creating static directory path for: " + self.STATIC_DIR)
        os.makedirs(self.STATIC_DIR, exist_ok=True)




