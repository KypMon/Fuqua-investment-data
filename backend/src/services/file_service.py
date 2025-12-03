import os
import threading
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

        self.is_local_host = True if config.get("HOST") == "localhost" else False

        #self.STATIC_DIR = config.get("static.dir")
        # this is for user file uploads / downloads
        self._token_map = {}
        self._lock = threading.Lock()

    def register_user_file(self, user, token: str, file_path: str):
        """
        Map a one-time token to this user's file path.
        You could attach expiry or periodic cleanup if desired.
        """
        # user_id = getattr(user, "userid", None) or user.get_userid()
        user_id = getattr(user, "user_id", None) or ""

        self.logger.info("Generating file token for user " +str(user_id) + " and file " + str(file_path) + " ...")

        with self._lock:
            self._token_map[token] = {"user_id": user_id, "path": file_path}
            self.show_token_map()

    def resolve_user_token(self, user, token: str):
        user_id = getattr(user, "user_id", None) or ""

        """Return the file info *iff* this user owns that token."""
        self.logger.info("Resolving file token " + token + " for user " +str(user_id) + " ...")
        self.show_token_map()

        with self._lock:
            entry = self._token_map.get(token)
            if not entry:
                self.logger.warning("No token")
                return None
            if entry["user_id"] != user_id:
                self.logger.warning("shenanigans")
                return None
            
            #self.logger.info("returning token entry: " + str(entry))
            return entry
        
    def show_token_map(self):
        self.logger.info("Current token map:")
        for token,token_dict in self._token_map.items():
            for k,v in token_dict.items():
                self.logger.info("File token: " + token + " -> " + k + " -> " + str(v))

    def get_etf_file_path(self) -> str:
        return self.etf_file
    
    def get_ff_file_path(self) -> str:
        return self.ff_file
    
    def get_ff5_file_path(self) -> str:
        return self.ff5_file
    
    def get_mom_file_path(self) -> str:
        return self.mom_file
    
    # def get_STATIC_DIR(self) -> str:
    #     return self.STATIC_DIR
    
    # def save_dataframe(self, user, df: pd.DataFrame, prefix: str) -> str:
    #     filename = self.timestamped_filename(prefix)
    #     if self.is_local_host is False:
    #         filename = self.user_timestamped_filename(filename, user) # embed userId
    #     path = os.path.join(self.STATIC_DIR, filename)
    #     #self.logger.info("Saving " + path + " to dataframe")
    #     df.to_csv(path, index=False)
    #     return filename

    def save_dataframe(self, user, df: pd.DataFrame, prefix: str, static_dir: str) -> str:
        filename = self.timestamped_filename(prefix)
        if self.is_local_host is False:
            filename = self.user_timestamped_filename(filename, user) # embed userId

        # path = os.path.join(self.STATIC_DIR, filename)
        path = os.path.join(static_dir, filename)
        #self.logger.info("Saving " + path + " to dataframe")
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




