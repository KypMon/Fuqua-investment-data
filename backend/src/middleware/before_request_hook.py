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
            static = request.environ.get("static")
            if static is True:
                return
            
            options = request.environ.get("options")
            if options is True:
                return

            status_code = request.environ.get("status_code")
            if status_code == 401:
                self.logger.info("HTTP status code is " + str(status_code))
                login_url = Config.get_property("fw.login.url")
                home_page_redirect = Config.get_property("home.page.redirect")
                self.logger.info("FuquaWorld login redirect is: " + login_url + home_page_redirect)
                return redirect(login_url + home_page_redirect)
            
            if status_code == 500:
                self.logger.info("HTTP status code is " + str(status_code))
                
                error_message = request.environ.get("err")
                #return render_template('500.html',error_message=error_message)
                self.logger.error("500 status code")
            
            # verify user is authorized to use the application
            fwUser=FwUser(request.environ.get("claims"))
            if fwUser is None:
                self.logger.error("FwUser is None")

            # make available to all routes
            g.fwUser = fwUser