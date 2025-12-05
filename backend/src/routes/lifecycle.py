import os
from uuid import uuid4
from flask import Blueprint, request, jsonify, g, url_for, abort, send_file
from datetime import datetime
from werkzeug.utils import secure_filename
from src.middleware.fw_user import FwUser
from src.services.life_cycle_service import LifeCycleService
from src.services.life_cycle_input_error import LifeCycleInputError
from src.services.file_service import FileService
from src.services.utilities_service import UtilitiesService
from src.logging.app_logger import AppLogger

class LifeCycle(object):

    def __init__(self, app=None) -> None:
        self.logger = AppLogger.get_logger()
        self.utilities_service = UtilitiesService()
        self.file_service = FileService()
        self.life_cycle_service = LifeCycleService()

        self.APP_PREFIX = os.getenv("APP_PREFIX", "")  # "/financial_analyzer" or ""
        self.logger.info("APP_PREFIX: " + self.APP_PREFIX)

        # determine absolute path for file uploads/downloads
        server_static_dir = os.getenv("STATIC_DIR")
        if server_static_dir: # server environment
            static_dir = server_static_dir + "/lifecycle" # /static/lifecycle
        else: # localhost environment
            module_dir = os.path.dirname(os.path.abspath(__file__))
            static_dir = os.path.join(module_dir, "static")

        token_dir = os.path.join(static_dir, "tokens")

        # guarantee folder exists
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(token_dir, exist_ok=True)

        self.static_dir = static_dir
        self.token_dir = token_dir

        self.blueprint = Blueprint(
            "LifeCycle",
            __name__,
            url_prefix=f"{self.APP_PREFIX}/lifecycle",
            static_url_path="/static",     # served at /lifecycle/static
            static_folder=static_dir,
        )

        # self.blueprint.add_url_rule(
        #     "/matret/download/<token>",
        #     view_func = self.download_matret,
        #     methods=["GET"],
        # )

        self.blueprint.add_url_rule(
            "/life-cycle/run",
            view_func=self.run_life_cycle,
            methods=["POST"],
        )

        if app:
            app.register_blueprint(self.blueprint)

    #@bp.route(f"{self.APP_PREFIX}/life-cycle/run", methods=["POST"])
    def run_life_cycle(self):
        self.utilities_service.log_user_activity()

        try:
            returns_file = request.files.get("returns_file")
            cashflows_file = request.files.get("cashflows_file")

            returns_vector = self.life_cycle_service.load_vector_from_csv(returns_file, "Return")
            cashflow_vector = self.life_cycle_service.load_vector_from_csv(cashflows_file, "Cash flow")

            form_data = request.form or {}
            if not form_data:
                form_data = request.json or {}

            initial_wealth = self._parse_float(form_data.get("initial_wealth", 0), "Initial wealth", 0.0)
            wmin_cutoff = self._parse_float(form_data.get("wmin_cutoff", 0), "Minimum wealth cutoff", 0.0)
            nsim = self._parse_int(form_data.get("nsim", 1000), "Number of simulations", 1000)

            result = self.life_cycle_service.run_life_cycle_analysis(
                returns_vector,
                cashflow_vector,
                w0=initial_wealth,
                wmin_cutoff=wmin_cutoff,
                nsim=nsim,
            )

            return jsonify(result)

        except LifeCycleInputError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - defensive fallback
            import traceback

            traceback.print_exc()
            return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500
        
    def _parse_float(self, value, label, default=0.0):
        if value in (None, ""):
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            raise LifeCycleInputError(f"{label} must be a numeric value.")


    def _parse_int(self, value, label, default=0):
        if value in (None, ""):
            return int(default)
        try:
            return int(float(value))
        except (TypeError, ValueError):
            raise LifeCycleInputError(f"{label} must be an integer value.")
    
    def build_download_url_via_token(self, fwUser, file_path, filename):
        # --- NEW: Generate unguessable token and store mapping ---
        token = uuid4().hex
        self.file_service.register_user_file(fwUser, token, file_path, self.token_dir)

        # Build URL to download via token
        # download_url = url_for("Matrix.download_matret", token=token)
        download_url = (url_for("Matrix.download_matret", token=token)).replace(self.APP_PREFIX, "")
        return download_url
    
    def get_blueprint(self):
        return LifeCycle.blueprint