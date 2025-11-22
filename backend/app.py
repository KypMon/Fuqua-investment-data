from flask import Flask
import matplotlib.pyplot as plt
import os

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
    config_file = config_file or os.environ.get("APP_CONFIG_FILE", "config/.env")
    log_file = log_file or os.environ.get("APP_LOG_FILE", "app.log")

    AppLogger.set_up_logger(log_file)
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

# if __name__ == "__main__": 
#     # for local development ("python app.py ....")
#     app = create_app()

#     app.run(
#         host=Config.get_property("HOST"),
#         port=Config.get_property("PORT"),
#         debug=False
#     )
# else:  # For gunicorn server threads 
#     print("FOR Gunicorn?")
#     app = create_app()

app = create_app()
if __name__ == "__main__": 
    # have Flask serve resources
    # for local development ("python app.py ....")
    app.run(
        host=Config.get_property("HOST"),
        port=Config.get_property("PORT"),
        debug=False
    )
else:
    # gunicorn is the server, no flask
    pass

