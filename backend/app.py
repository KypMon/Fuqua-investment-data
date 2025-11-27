from flask import Flask
from flask_cors import CORS
import os
from datetime import datetime
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from src.logging.app_logger import AppLogger
from src.config.config import Config
from src.middleware.authentication import Authentication
from src.middleware.before_request_hook import BeforeRequestHook
from src.routes.api_routes import ApiRoutes

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = "app.log_" + timestamp + ".log"
AppLogger.set_up_logger(log_file)

config_file = os.environ.get("APP_CONFIG_FILE", "config/.env")
Config.set_up_config(config_file)

def get_app():
    app = Flask(
        __name__,
        static_url_path="/static",
        static_folder=os.path.join(os.path.dirname(__file__), "static")
    )
    return app

def register_routes(app):
    app.register_blueprint(ApiRoutes().blueprint)

###
### when running on a Fuqua server
###
def create_app_server(config_file=None, log_file=None):
    app = get_app()
    #app.config["LOG_FILE"] = log_file
    #app.wsgi_app = Authentication(app.wsgi_app)
    app.wsgi_app = Authentication(DispatcherMiddleware(app.wsgi_app, {
        '/financial_analyzer': app.wsgi_app
    }))

    BeforeRequestHook().register_hooks(app)
    register_routes(app)
    return app

###
### when running under localhost
###
def create_app_localhost(config_file=None, log_file=None):
    app = get_app()

    CORS(app, supports_credentials=True, resources={
        r"/*": {
            "origins": ["http://localhost:3000"]
        }
    })

    STATIC_DIR = "static"
    os.makedirs(STATIC_DIR, exist_ok=True)
    register_routes(app)
    return app

if __name__ == "__main__": 
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #log_file = "app.log_" + timestamp + ".log"
    AppLogger.set_up_logger("app.log")

    config_file = os.environ.get("APP_CONFIG_FILE", "config/.env")
    Config.set_up_config(config_file)

    app = create_app_localhost()
    app.run(host="0.0.0.0", port=5001, debug=False
    )
else:
    log = AppLogger.get_logger()
    log.info("SERVER")
    # gunicorn is our server ...
    # see wsgi.py for gunicorn entry point
    app = create_app_server()
    log.info(str(Config.get_property("HOST")) + " " + str(Config.get_property("PORT")))
    app.run(host=Config.get_property("HOST"), port=int(Config.get_property("PORT")))

