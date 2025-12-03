import pandas as pd
import os
from uuid import uuid4
from flask import Blueprint, request, jsonify, g, url_for, abort, send_file
from datetime import datetime
from werkzeug.utils import secure_filename
from src.middleware.fw_user import FwUser
from src.services.matrix_service import MatrixService
from src.services.file_service import FileService
from src.logging.app_logger import AppLogger

class Matrix(object):

    def __init__(self, app=None) -> None:
        self.logger = AppLogger.get_logger()
        self.matrix_service = MatrixService()
        self.file_service = FileService()

        self.APP_PREFIX = os.getenv("APP_PREFIX", "")  # "/financial_analyzer" or ""

        # determine absolute path
        module_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join(module_dir, "static")
        token_dir = os.path.join(module_dir, "static/tokens")

        # guarantee folder exists
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(token_dir, exist_ok=True)

        self.static_dir = static_dir
        self.token_dir = token_dir

        self.blueprint = Blueprint(
            "Matrix",
            __name__,
            url_prefix=f"{self.APP_PREFIX}/matrix",
            static_url_path="/static",     # served at /matrix/static
            static_folder=static_dir,
        )

        self.blueprint.add_url_rule(
            "/matret/download/<token>",
            view_func = self.download_matret,
            methods=["GET"],
        )

        self.blueprint.add_url_rule(
            "/matret/generate",
            view_func=self.matrix_generate_matret,
            methods=["POST"],
        )

        self.blueprint.add_url_rule(
            "/matret/upload",
            view_func=self.matrix_upload_matret,
            methods=["POST"],
        )

        self.blueprint.add_url_rule(
            "/mat_er_covr/generate",
            view_func=self.matrix_generate_mat_er_covr,
            methods=["POST"],
        )

        self.blueprint.add_url_rule(
            "/mat_er_covr/upload",
            view_func=self.matrix_upload_mat_er_covr,
            methods=["POST"],
        )

        self.blueprint.add_url_rule(
            "/portfolios",
            view_func=self.matrix_compute_portfolios,
            methods=["POST"],
        )

        if app:
            app.register_blueprint(self.blueprint)

    # GENERATE MATRET
    def matrix_generate_matret(self):
        self.log_user_activity()

        data = request.json or {}

        tickers = data.get("tickers", [])
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]

        start_date = data.get("start_date", "2000-01-01")
        end_date = data.get("end_date", datetime.today().strftime("%Y-%m-%d"))

        try:
            matret_df, available = self.matrix_service.download_matret(
                tickers, start_date, end_date
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        fwUser = getattr(g, "fwUser", None)

        # Save dataframe and get a file path (still under your static_dir for now)
        filename = self.file_service.save_dataframe(fwUser, matret_df, "matret", self.static_dir)
        file_path = os.path.join(self.static_dir, filename)
        download_url = self.build_download_url_via_token(fwUser, file_path, filename)

        #self.logger.info("GENERATE filename: " + str(filename))
        #self.logger.info("GENERATE file_path: " + str(file_path))
        #self.logger.info("GENERATE download_url: " + str(download_url))

        return jsonify({
            "matrix": self.dataframe_payload(matret_df),
            "tickers": available,
            "csv_url": download_url
        })
    
    # Download latest matret CSV
    def download_matret(self, token):
        """
        Secure download endpoint. Uses a token issued by the generate route.
        Only the owner of the file can access it.
        """

        self.log_user_activity()
        user = getattr(g, "fwUser", None)

        # Ask FileService to look up and validate this token for the current user
        entry = self.file_service.resolve_user_token(user, token, self.token_dir)
        self.logger.info("DOWNLOAD LATEST: resolved user token is: " + str(entry))

        if not entry:
            # token not found or doesn't belong to this user
            self.logger.error("Aborting 403")
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
            self.logger.error("Aborting 404")
            abort(404)
    
    # UPLOAD MATRET CSV 
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

        user = getattr(g, "fwUser", None)
        filename = self.file_service.save_dataframe(user, df, "matret_upload", self.static_dir)
        file_path = os.path.join(self.static_dir, filename)
        download_url = self.build_download_url_via_token(user, file_path, filename)

        # self.logger.info("UPLOAD filename: " + str(filename))
        # self.logger.info("UPLOAD file_path: " + str(file_path))
        # self.logger.info("UPLOAD download_url: " + str(download_url))

        return jsonify(
            {
                "matrix": self.dataframe_payload(df),
                #"csv_url": f"/static/{filename}",
                #"csv_url": f"{self.blueprint.url_prefix}{self.blueprint.static_url_path}/{filename}",
                "csv_url": download_url,
                "original_filename": secure_filename(file.filename),
            }
        )

    # GENERATE MAT_ER_COVR
    #@blueprint.route("/matrix/mat_er_covr/generate", methods=["POST"])
    def matrix_generate_mat_er_covr(self):
        self.log_user_activity()

        data = request.json or {}

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

        fwUser = getattr(g, "fwUser", None)

        #filename = self.file_service.save_dataframe(fwUser, mat_er_covr_df, "mat_er_covr", self.static_dir)
        filename = self.file_service.save_dataframe(fwUser, mat_er_covr_df, "mat_er_covr", self.static_dir)
        file_path = os.path.join(self.static_dir, filename)
        download_url = self.build_download_url_via_token(fwUser, file_path, filename)

        self.logger.info("GENERATE_COVR filename: " + str(filename))
        self.logger.info("GENERATE_COVR file_path: " + str(file_path))
        self.logger.info("GENERATE_COVR download_url: " + str(download_url))

        return jsonify(
            {
                "matrix": self.dataframe_payload(mat_er_covr_df),
                "risk_free": resolved_rf,
                #"csv_url": f"/static/{filename}",
                # "csv_url": f"{self.blueprint.url_prefix}{self.blueprint.static_url_path}/{filename}",
                "csv_url": download_url
            }
        )

    # UPLOAD MAT_ER_COVR CSV
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

        user = getattr(g, "fwUser", None)
        filename = self.file_service.save_dataframe(user, df, "mat_er_covr_upload", self.static_dir)
        file_path = os.path.join(self.static_dir, filename)
        download_url = self.build_download_url_via_token(user, file_path, filename)

        # self.logger.info("UPLOAD filename: " + str(filename))
        # self.logger.info("UPLOAD file_path: " + str(file_path))
        # self.logger.info("UPLOAD download_url: " + str(download_url))

        return jsonify(
            {
                "matrix": self.dataframe_payload(df),
                "risk_free": rf_value,
                #"csv_url": f"/static/{filename}",
                #"csv_url": f"{self.blueprint.url_prefix}{self.blueprint.static_url_path}/{filename}",
                "csv_url": download_url,
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
    
    def build_download_url_via_token(self, fwUser, file_path, filename):
        # --- NEW: Generate unguessable token and store mapping ---
        token = uuid4().hex
        self.file_service.register_user_file(fwUser, token, file_path, self.token_dir)

        # Build URL to download via token
        # download_url = url_for("Matrix.download_matret", token=token)
        download_url = (url_for("Matrix.download_matret", token=token)).replace(self.APP_PREFIX, "")
        return download_url
    
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