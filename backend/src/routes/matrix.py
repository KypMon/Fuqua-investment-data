import pandas as pd
import os
from flask import Blueprint, request, jsonify, g
from datetime import datetime
from werkzeug.utils import secure_filename
from src.middleware.fw_user import FwUser
from src.services.matrix_service import MatrixService
from src.services.file_service import FileService
from src.logging.app_logger import AppLogger

class Matrix(object):

    def __init__(self, app=None) -> None:
        self.logger = AppLogger.get_logger()
        self.logger.info("This is Matrix constructor")
        self.matrix_service = MatrixService()
        self.file_service = FileService()

        self.APP_PREFIX = os.getenv("APP_PREFIX", "")  # "/financial_analyzer" or ""
        self.logger.info("self.APP_PREFIX: " + str(self.APP_PREFIX))

        # determine absolute path
        module_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join(module_dir, "static")
        #self.logger.info(str(static_dir))

        # guarantee folder exists
        os.makedirs(static_dir, exist_ok=True)

        self.static_dir = static_dir

        self.blueprint = Blueprint(
            "Matrix",
            __name__,
            url_prefix="/matrix",
            static_url_path="/static",     # served at /matrix/static
            static_folder=static_dir,
        )

        self.blueprint.add_url_rule(
            #"/matret/generate",
            f"{self.APP_PREFIX}/matret/generate", 
            view_func=self.matrix_generate_matret,
            methods=["POST"],
        )

        self.blueprint.add_url_rule(
            "/matret/upload",
            #f"{self.APP_PREFIX}/matret/upload", 
            view_func=self.matrix_upload_matret,
            methods=["POST"],
        )

        self.blueprint.add_url_rule(
            "/mat_er_covr/generate",
            #f"{self.APP_PREFIX}/mat_er_covr/generate", 
            view_func=self.matrix_generate_mat_er_covr,
            methods=["POST"],
        )

        self.blueprint.add_url_rule(
            "/mat_er_covr/upload",
            #f"{self.APP_PREFIX}/mat_er_covr/upload", 
            view_func=self.matrix_upload_mat_er_covr,
            methods=["POST"],
        )

        self.blueprint.add_url_rule(
            "/portfolios",
            #f"{self.APP_PREFIX}/portfolios", 
            view_func=self.matrix_compute_portfolios,
            methods=["POST"],
        )

        if app:
            app.register_blueprint(self.blueprint)

    #@blueprint.route("/matrix/matret/generate", methods=["POST"])
    def matrix_generate_matret(self):
        data = request.json or {}
        self.log_user_activity()

        tickers = data.get("tickers", [])
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]

        start_date = data.get("start_date", "2000-01-01")
        end_date = data.get("end_date", datetime.today().strftime("%Y-%m-%d"))

        try:
            matret_df, available = self.matrix_service.download_matret(tickers, start_date, end_date)
        except Exception as exc:  # pragma: no cover - defensive error path
            return jsonify({"error": str(exc)}), 400

        # filename = save_dataframe(matret_df, "matret")
        user = getattr(g, "fwUser", None)
        filename = self.file_service.save_dataframe(user, matret_df, "matret", self.static_dir)
        return jsonify(
            {
                "matrix": self.dataframe_payload(matret_df),
                "tickers": available,
                # "csv_url": f"/static/{filename}",
                #"csv_url": f"/routes/static/{filename}",
                "csv_url": f"{self.blueprint.url_prefix}{self.blueprint.static_url_path}/{filename}"
            }
        )

    #@blueprint.route("/matrix/matret/upload", methods=["POST"])
    def matrix_upload_matret(self):
        self.log_user_activity()
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        try:
            df = pd.read_csv(file)
        except Exception as exc:  # pragma: no cover - pandas error path
            return jsonify({"error": f"Unable to read CSV: {exc}"}), 400

        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        # @SDS begin
        # filename = save_dataframe(df, "matret_upload")
        user = getattr(g, "fwUser", None)
        filename = self.file_service.save_dataframe(user, df, "matret_upload", self.static_dir)
        # @SDS end

        return jsonify(
            {
                "matrix": self.dataframe_payload(df),
                #"csv_url": f"/static/{filename}",
                 "csv_url": f"{self.blueprint.url_prefix}{self.blueprint.static_url_path}/{filename}",
                "original_filename": secure_filename(file.filename),
            }
        )

    #@blueprint.route("/matrix/mat_er_covr/generate", methods=["POST"])
    def matrix_generate_mat_er_covr(self):
        data = request.json or {}
        self.log_user_activity()

        matret_payload = data.get("matret")
        risk_free = data.get("risk_free")

        try:
            matret_df = self.matrix_service.parse_matrix_payload(matret_payload)
        except Exception as exc:
            return jsonify({"error": f"Invalid matret payload: {exc}"}), 400

        try:
            rf_value = float(risk_free) if risk_free is not None else None
        except (TypeError, ValueError):
            return jsonify({"error": "Risk-free rate must be numeric"}), 400

        try:
            mat_er_covr_df, resolved_rf = self.matrix_service.create_mat_er_covr(matret_df, rf_value)
        except Exception as exc:  # pragma: no cover - numeric errors
            return jsonify({"error": str(exc)}), 400

        # @SDS begin
        # filename = save_dataframe(mat_er_covr_df, "mat_er_covr")
        user = getattr(g, "fwUser", None)
        filename = self.file_service.save_dataframe(user, mat_er_covr_df, "mat_er_covr", self.static_dir)
        # @SDS end
        return jsonify(
            {
                "matrix": self.dataframe_payload(mat_er_covr_df),
                "risk_free": resolved_rf,
                #"csv_url": f"/static/{filename}",
                 "csv_url": f"{self.blueprint.url_prefix}{self.blueprint.static_url_path}/{filename}",
            }
        )

    #@blueprint.route("/matrix/mat_er_covr/upload", methods=["POST"])
    def matrix_upload_mat_er_covr(self):
        self.log_user_activity()

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        try:
            df = pd.read_csv(file)
        except Exception as exc:
            return jsonify({"error": f"Unable to read CSV: {exc}"}), 400

        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        rf_value = None
        if "Assets" in df.columns and "Mean" in df.columns:
            asset_series = df["Assets"].astype(str).str.lower()
            rf_rows = df[asset_series == "rf"]
            if not rf_rows.empty:
                numeric_rf = pd.to_numeric(rf_rows["Mean"], errors="coerce").dropna()
                if not numeric_rf.empty:
                    rf_value = float(numeric_rf.iloc[0])

        # @SDS begin
        # filename = save_dataframe(df, "mat_er_covr_upload")
        user = getattr(g, "fwUser", None)
        filename = self.file_service.save_dataframe(user, df, "mat_er_covr_upload", self.static_dir)
        # @ SDS end
        return jsonify(
            {
                "matrix": self.dataframe_payload(df),
                "risk_free": rf_value,
                #"csv_url": f"/static/{filename}",
                 "csv_url": f"{self.blueprint.url_prefix}{self.blueprint.static_url_path}/{filename}",
                "original_filename": secure_filename(file.filename),
            }
        )

    #@blueprint.route("/matrix/portfolios", methods=["POST"])
    def matrix_compute_portfolios(self):
        data = request.json or {}
        self.log_user_activity(data)

        mat_er_covr_payload = data.get("mat_er_covr")
        risk_free = data.get("risk_free")

        try:
            mat_er_covr_df = self.matrix_service.parse_matrix_payload(mat_er_covr_payload)
        except Exception as exc:
            return jsonify({"error": f"Invalid mat_er_covr payload: {exc}"}), 400

        try:
            rf_value = float(risk_free) if risk_free is not None else None
        except (TypeError, ValueError):
            return jsonify({"error": "Risk-free rate must be numeric"}), 400

        print(rf_value)

        try:
            result = self.matrix_service.compute_portfolios(mat_er_covr_df, rf_value)
        except Exception as exc:  # pragma: no cover - numeric errors
            print(exc)
            return jsonify({"error": f"Compute error: {str(exc)}"}), 400

        return jsonify(result)
    
    def dataframe_payload(self, df: pd.DataFrame) -> dict:
        """Serialize a dataframe for JSON responses.

        ``pandas`` represents missing values as ``NaN`` which does not have a
        native JSON representation.  ``json.dumps`` would emit the JavaScript
        identifier ``NaN`` which is invalid JSON and causes ``JSON.parse`` to
        throw on the frontend.  To keep the payloads consumable for both uploaded
        and generated matrices we normalise the dataframe and replace missing
        values with ``None`` (rendered as ``null`` in JSON) before converting it to
        dictionaries.
        """

        sanitized = df.copy().astype(object)
        sanitized = sanitized.where(pd.notna(sanitized), None)
        return {
            "columns": list(sanitized.columns),
            "records": sanitized.to_dict(orient="records"),
        }
    
    def log_user_activity(self, data=None):
        if not (
            request.url.startswith("/static/")
            or request.url.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".map"))
        ):
            user = getattr(g, "fwUser", None)
            if not user is None:
                self.logger.info(user.get_dukeid() + " " + user.get_userid() + " " + user.get_name() + " -> " + request.url + " " + (str(data) if data is not None else ""))
            else:
                self.logger.info(request.url + " " + (str(data) if data is not None else ""))

    
    def get_blueprint(self):
        return Matrix.blueprint