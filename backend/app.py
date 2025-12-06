from flask import Flask
from flask_cors import CORS
import os
from datetime import datetime
from src.logging.app_logger import AppLogger
from src.config.config import Config
from src.middleware.authentication import Authentication
from src.middleware.before_request_hook import BeforeRequestHook
from src.routes.api_routes import ApiRoutes
from src.routes.backtest import Backtest
from src.routes.regression import Regression
from src.routes.matrix import Matrix
from src.routes.lifecycle import LifeCycle

def get_app():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    app = Flask(
        __name__,
        static_url_path="/static",
        static_folder=os.path.join(BASE_DIR, Config.get_property("react.build.dir"), "static")
    )
    return app

def register_routes(app):
    app.register_blueprint(ApiRoutes().blueprint)
    Backtest(app)
    Regression(app)
    Matrix(app)
    LifeCycle(app)

### ##############################
### when running on a Fuqua server
### ##############################
def create_app_server(config_file=None, log_file=None):
    app = get_app()
    app.wsgi_app = Authentication(app.wsgi_app)

    BeforeRequestHook().register_hooks(app)
    register_routes(app)
    return app

### #############################
### when running under localhost
### #############################
def create_app_localhost(config_file=None, log_file=None):
    app = get_app()

    CORS(app, supports_credentials=True, resources={
        r"/*": {
            "origins": ["http://localhost:3000"]
        }
    })

    register_routes(app)
    return app

if __name__ == "__main__": # only runs if executing "python app.py" in localhost environment
    ###
    ### Flask entry point for localhost:3000 (single-user, no user authentication)
    ###
    AppLogger.set_up_logger("app.log")
    log = AppLogger.get_logger()

    config_file = os.environ.get("APP_CONFIG_FILE", "config/.env")
    Config.set_up_config(config_file)

    app = create_app_localhost()
    app.run(host="0.0.0.0", port=5001, debug=False
    )
else:
    ###
    ### see wsgi.py for gunicorn entry point (multi-user, users must authenticate)
    ### 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = "app.log_" + timestamp + ".log"
    AppLogger.set_up_logger(log_file)
    log = AppLogger.get_logger()

    config_file = os.environ.get("APP_CONFIG_FILE", "config/.env")
    Config.set_up_config(config_file)

    app = create_app_server()
    # gunicorn is our server, so we do not say 'app.run' 
    #app.run(host=Config.get_property("HOST"), port=int(Config.get_property("PORT")))

