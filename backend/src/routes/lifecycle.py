import os
import pandas as pd
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

        # register download endpoint for summary files
        self.blueprint.add_url_rule(
            "/life-cycle/download/<token>",
            view_func=self.download_life_cycle_summary,
            methods=["GET"],
        )

        self.blueprint.add_url_rule(
            "/life-cycle/run",
            view_func=self.run_life_cycle,
            methods=["POST"],
        )

        if app:
            app.register_blueprint(self.blueprint)

    def run_life_cycle(self):
        self.utilities_service.log_user_activity()

        user = getattr(g, "fwUser", None)

        try:
            returns_file = request.files.get("returns_file")
            cashflows_file = request.files.get("cashflows_file")

            form_data = request.form or {}
            if not form_data:
                form_data = request.json or {}

            initial_wealth = self._parse_float(form_data.get("initial_wealth", 0), "Initial wealth", 0.0)
            wmin_cutoff = self._parse_float(form_data.get("wmin_cutoff", 0), "Minimum wealth cutoff", 0.0)
            nsim = self._parse_int(form_data.get("nsim", 1000), "Number of simulations", 1000)

            # Read both files into dataframes
            #df_returns = pd.read_csv(returns_file)
            #df_cashflows = pd.read_csv(cashflows_file)

            # Reset file pointer for reuse
            #returns_file.seek(0)
            #cashflows_file.seek(0)

            # Save them with timestamp + userid 
            #returns_filename = self.file_service.save_dataframe(user, df_returns, "life_cycle_returns", self.static_dir)
            #cashflows_filename = self.file_service.save_dataframe(user, df_cashflows, "life_cycle_cashflows", self.static_dir)

            # run the simulation
            returns_vector = self.life_cycle_service.load_vector_from_csv(returns_file, "Return")
            cashflow_vector = self.life_cycle_service.load_vector_from_csv(cashflows_file, "Cash flow")

            result = self.life_cycle_service.run_life_cycle_analysis(
                returns_vector,
                cashflow_vector,
                w0=initial_wealth,
                wmin_cutoff=wmin_cutoff,
                nsim=nsim,
            )

            # Save summary CSV (use same helper)
            summary_df = self.life_cycle_service.to_summary_dataframe(result)
            summary_filename = self.file_service.save_dataframe(user, summary_df, "life_cycle_summary", self.static_dir)
            summary_filepath = os.path.join(self.static_dir, summary_filename)
            download_url = self.build_download_url_via_token(user, summary_filepath, summary_filename)

            # Register for secure token download
            #summary_path = os.path.join(self.static_dir, summary_filename)
            #csv_url = self.build_download_url_via_token(user, summary_path, summary_filename)
            self.logger.info("summary_filename: " + str(summary_filename))
            self.logger.info("summary_filepath: " + str(summary_filepath))
            self.logger.info("download url: " + str(download_url))

            return jsonify({
                **result,  # unpack the keys inside result dict
                "summary_csv_url": download_url,
                #"returns_filename": returns_filename,    
                #"cashflows_filename": cashflows_filename,
            })

        except LifeCycleInputError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - defensive fallback
            import traceback

            traceback.print_exc()
            return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500
        
    #@bp.route("/life-cycle/download/<token>")
    def download_life_cycle_summary(self, token):
        self.logger.info("DOWNLOAD LIFE CYCLE SUMMARY")
        self.utilities_service.log_user_activity()
        user = getattr(g, "fwUser", None)

        entry = self.file_service.resolve_user_token(user, token, self.token_dir)
        if not entry:
            abort(403)

        file_path = entry["path"]
        try:
            return send_file(
                file_path,
                as_attachment=True,
                download_name=os.path.basename(file_path),
                mimetype="text/csv",
                max_age=0,
                conditional=False
            )
        except FileNotFoundError:
            abort(404)
        
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
        download_url = (url_for("LifeCycle.download_life_cycle_summary", token=token)).replace(self.APP_PREFIX, "")
        return download_url
    
    def get_blueprint(self):
        return LifeCycle.blueprint