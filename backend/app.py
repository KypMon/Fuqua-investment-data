from flask import Flask
from flask_cors import CORS
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
def create_app_server(config_file=None, log_file=None):

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

def create_app_localhost(config_file=None, log_file=None):

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
    #CORS(app, resources={r"/*": {"origins": "*"}})
    CORS(app, supports_credentials=True, resources={
        r"/*": {
            "origins": ["http://localhost:3000"]
        }
    })
    STATIC_DIR = "static"
    os.makedirs(STATIC_DIR, exist_ok=True)

    app.register_blueprint(ApiRoutes().blueprint)

    return app

#app = create_app()

if __name__ == "__main__": 
    print(__name__)
    app = create_app_localhost()
    app.run(host="0.0.0.0", port=5001, debug=False
    )
else:
    print(__name__)
    # gunicorn is our server ...
    # see wsgi.py for gunicorn entry point
    # (for running this app on Linux servers in Fuqua domain somewhere)
    pass

