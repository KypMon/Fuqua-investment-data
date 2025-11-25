from flask import Flask
import matplotlib.pyplot as plt
import os
from datetime import datetime
from src.logging.app_logger import AppLogger
from src.config.config import Config
from src.services.file_service import FileService
from src.middleware.authentication import Authentication
from src.middleware.before_request_hook import BeforeRequestHook
from src.routes.api_routes import ApiRoutes

"""
Application factory: used both by Flask (dev) and Gunicorn (prod)
"""
def create_app(config_file=None, log_file=None):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = "app.log_" + timestamp + ".log"
    AppLogger.set_up_logger(log_file)

    config_file = config_file or os.environ.get("APP_CONFIG_FILE", "config/.env")
    Config.set_up_config(config_file)

    app = Flask(
        __name__,
        static_url_path="/static",
        static_folder=os.path.join(os.path.dirname(__file__), "static")
    )

    app.config["LOG_FILE"] = log_file

    app.wsgi_app = Authentication(app.wsgi_app)

    BeforeRequestHook().register_hooks(app)

    app.register_blueprint(ApiRoutes().blueprint)

    return app

app = create_app()

if __name__ == "__main__": 
    # have Flask serve backend resources
    # for local development ("python app.py" and "localhost:5002")
    FileService().make_static_dir()

    app.run(
        host=Config.get_property("HOST"),
        port=Config.get_property("PORT"),
        debug=False
    )
else:
    # gunicorn is our server ...
    # see wsgi.py for gunicorn entry point
    # (for running this app on Linux servers in Fuqua domain somewhere)
    pass

