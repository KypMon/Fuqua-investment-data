# gunicorn.conf.py
import os
from src.config.config import Config

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_LOG = os.path.join(_ROOT, 'financial_analyzer')

loglevel = "info"
errorlog = os.path.join(_LOG, 'log/financial_analyzer.log')
accesslog = os.path.join(_LOG, 'log/gunicorn-api-access.log')

bind = Config.get_property("HOST") + ":" + Config.get_property("PORT")

# workers = multiprocessing.cpu_count() * 2 + 1
workers = 4

timeout = 3 * 60 # 3 minutes
keepalive = 24 * 60 * 60 # 1 day

capture_output = True

wsgi_app = "wsgi:app"
