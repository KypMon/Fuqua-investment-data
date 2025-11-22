from flask import Flask
import matplotlib.pyplot as plt
import os

from src.logging.app_logger import AppLogger
from src.config.config import Config
from src.services.file_service import FileService
from src.middleware.authentication import Authentication
from src.middleware.before_request_hook import BeforeRequestHook
from src.routes.api_routes import ApiRoutes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str)
parser.add_argument("--config", type=str)

args = parser.parse_args()
log = AppLogger.set_up_logger(args.log)
config = Config.set_up_config(args.config)

def create_app(config_file=None, log_file=None):
    """
    Application factory: used both by Flask (dev) and Gunicorn (prod)
    """
    config_file = config_file or os.environ.get("APP_CONFIG_FILE", "config/.env")
    log_file = log_file or os.environ.get("APP_LOG_FILE", "app.log")

    Config.set_up_config(config_file)

    FileService().make_static_dir()
    
    app = Flask(
        __name__,
        static_url_path="/static",
        static_folder=os.path.join(os.path.dirname(__file__), "static")
    )

    app.config["LOG_FILE"] = log_file

    app.wsgi_app = Authentication(app.wsgi_app)

    BeforeRequestHook().register_hooks(app)

    api_routes = ApiRoutes()
    app.register_blueprint(api_routes.blueprint)

    return app

if __name__ == "__main__": 
    # for local development ("python app.py ....")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file", default="config/.env")
    parser.add_argument("--log", dest="log_file", default="app.log")
    args = parser.parse_args()

    app = create_app(config_file=args.config_file, log_file=args.log_file)

    app.run(
        host=Config.get_property("HOST"),
        port=Config.get_property("PORT"),
        debug=False
    )
else:  # For gunicorn server threads 
    app = create_app()
