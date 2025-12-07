from app import app
from src.config.config import Config
##
## This file serves as the entry point for gunicorn.
# If running in a non-localhost environment, use gunicorn to serve the app, not Flask.
#
# gunicorn --workers=1 --bind localhost.fuqua.duke.edu:5002 --access-logfile gunicorn.log wsgi:app
if __name__ == "__main__":
    #print("wsgi.py HOST: " + str(Config.get_property("HOST")))
    #print("wsgi.py PORT: " + str(Config.get_property("PORT")))
    app.run(host=Config.get_property("HOST"), port=Config.get_property("PORT"))
