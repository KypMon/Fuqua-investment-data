from flask import request, redirect, g
from src.config.config import Config
from src.logging.app_logger import AppLogger
from src.middleware.fw_user import FwUser

class BeforeRequestHook(object):
    def __init__(self):
        self.logger = AppLogger.get_logger()

    ##
    ## This is Flask's before_request decorator pattern.  
    ## Every Flask request is intercepted here.
    ##
    def register_hooks(self, app):
        @app.before_request
        def authenticate_and_authorize():
            status_code = request.environ.get("status_code")
            self.logger.info("BeforeRequestHook status_code: " + str(status_code))
            if status_code == 401:
                self.logger.info("HTTP status code is " + str(status_code))
                login_url = Config.get_property("fw.login.url")
                home_page_redirect = Config.get_property("home.page.redirect")
                self.logger.info(str(status_code) + ": Redirecting to: " + login_url + home_page_redirect)
                return redirect(login_url + home_page_redirect)
            
            if status_code == 500:
                err = request.environ.get("err")
                self.logger.info("HTTP status code is " + str(status_code) + " " + str(err))

            # verify user is authorized to use the application
            fwUser=FwUser(request.environ.get("claims"))
            if fwUser is None:
                self.logger.error("FwUser is None")

            # make available to all routes
            g.fwUser = fwUser