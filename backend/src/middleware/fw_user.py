from typing import Any
from src.logging.app_logger import AppLogger

class FwUser(object):
    def __init__(self, claims:dict) -> None:
        self.logger = AppLogger.get_logger()

        if claims is None:
            #self.logger.info("uh oh, no claims")
            return
        
        self.claims = claims 
        #self.logger.info(str(self.claims))
        #self.logger.info("Fuqua Finance Analyzer user: "  + self.claims["name"])

    def get_claims(self) -> dict:
        return self.claims

    def get_attribute(self, key:str) -> Any:
        return self.claims.get(key) if self.claims is not None else None

    def get_dukeid(self) -> str:
        return self.claims.get("dukeid") if self.claims is not None else None
    
    def get_userid(self) -> str:
        return self.claims.get("uid") if self.claims is not None else None
    
    def get_email(self) -> str:
        return self.claims.get("email") if self.claims is not None else None
    
    def get_name(self) -> str:
        return self.claims.get("name") if self.claims is not None else None
    
    def get_first_name(self) -> str:
        return self.claims.get("first") if self.claims is not None else None
    
    def get_last_name(self) -> str:
        return self.claims.get("last") if self.claims is not None else None
    
    def get_image_url(self) -> str:
        return self.claims.get("image") if self.claims is not None else None
    
    def get_canvasid(self) -> str:
        return self.claims.get("canvas") if self.claims is not None else None
    
    def get_fid(self) -> str:
        return self.claims.get("fid") if self.claims is not None else None
    
    def get_title(self) -> str:
        return self.claims.get("title") if self.claims is not None else None
    
    def get_object_classes(self) -> list:
        return self.claims.get("objectclasses") if self.claims is not None else None
