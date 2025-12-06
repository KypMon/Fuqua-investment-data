import os
import json
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

    def register_user_file(self, user, token, file_path, token_dir):
        """Store mapping on disk: one small JSON per token."""
        user_id = "" if user is None else user.get_userid()

        token_path = os.path.join(token_dir, f"{token}.json")
        self.logger.info("Registering user file for " + user_id + " at " + str(token_path))
        payload = {"user_id": user_id, "path": file_path}
        #self.logger.info("payload: " + str(payload))

        with open(token_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        #self.logger.info(f"Token file written for: {token_path}")

    def resolve_user_token(self, user, token, token_dir):
        user_id = "" if user is None else user.get_userid()

        """Read mapping back from disk and validate ownership."""
        token_path = os.path.join(token_dir, f"{token}.json")

        self.logger.info("Resolving user token file for " + user_id + " at " + str(token_path))

        if not os.path.exists(token_path):
            self.logger.warning(f"No such token file: {token_path}")
            return None

        try:
            with open(token_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            self.logger.error(f"Corrupt token file: {token_path}")
            return None

       
        if data.get("user_id") != user_id:
            self.logger.warning("Token mismatch (different user)")
            return None
        return data
    
    def get_etf_file_path(self) -> str:
        return self.etf_file
    
    def get_ff_file_path(self) -> str:
        return self.ff_file
    
    def get_ff5_file_path(self) -> str:
        return self.ff5_file
    
    def get_mom_file_path(self) -> str:
        return self.mom_file
    
    def save_dataframe(self, user, df: pd.DataFrame, prefix: str, static_dir: str) -> str:
        filename = self.timestamped_filename(prefix)
        if self.is_local_host is False:
            filename = self.user_timestamped_filename(filename, user) # embed userId

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
    
    def save_html(self, user, html_content: str, prefix: str, static_dir: str) -> str:
        filename = self.timestamped_filename(prefix).replace(".csv", ".html")
        if not self.is_local_host:
            filename = self.user_timestamped_filename(filename, user)
        path = os.path.join(static_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return filename
    

