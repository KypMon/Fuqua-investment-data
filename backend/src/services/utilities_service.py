from flask import  request, g, url_for
from uuid import uuid4
from src.logging.app_logger import AppLogger

class UtilitiesService(object):

    def __init__(self):
        self.logger = AppLogger.get_logger()

    #
    # can only call this method from within the context of a call to an API endpoint,
    # otherwise, user data will not be available
    #
    def log_user_activity(self, data=None):
        if not (
            request.url.startswith("/static/")
            or request.url.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".map"))
        ):
            user = getattr(g, "fwUser", None)
            if not user is None:
                self.logger.info(user.get_dukeid() + " " + user.get_userid() + " " + user.get_name() + " -> " + request.url + " " + (str(data) if data is not None else ""))
            else:
                self.logger.info(request.url + " " + (str(data) if data is not None else ""))

    def build_download_url_via_token(self, fwUser, file_path, urlFor):
        # --- NEW: Generate unguessable token and store mapping ---
        token = uuid4().hex
        self.file_service.register_user_file(fwUser, token, file_path, self.token_dir)

        # Build URL to download via token
        download_url = (url_for(urlFor, token=token)).replace(self.APP_PREFIX, "")
        return download_url

    
