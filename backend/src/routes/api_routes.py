#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
# import io
import os
# import statsmodels.api as sm
#from contextlib import redirect_stdout
#from statsmodels.stats.stattools import durbin_watson, jarque_bera
#from datetime import datetime
from typing import Any
from flask import Blueprint, jsonify, request, g, send_from_directory, redirect
from mv import mv
from src.logging.app_logger import AppLogger
from src.config.config import Config
from src.services.data_service import DataService
from src.services.file_service import FileService
from src.services.backtest_service import BacktestService
#from src.services.backtest_input_error import BacktestInputError
from src.services.utilities_service import UtilitiesService

class ApiRoutes(object):

    def __init__(self) -> None:
        self.logger = AppLogger.get_logger()

        self.APP_PREFIX = os.getenv("APP_PREFIX", "")  # "/financial_analyzer" or ""

        routes_dir = os.path.dirname(__file__)
        backend_root = os.path.abspath(os.path.join(routes_dir, "..", ".."))  # up to backend
        self.REACT_BUILD_PATH = os.path.join(backend_root, Config.get_property("react.build.dir"))

        self.blueprint = Blueprint("ApiRoutes", __name__)
        self._add_routes()

        self.utilities_service = UtilitiesService()
        self.data_service = DataService()
        self.file_service = FileService()
        self.backtest_service = BacktestService()

    def _add_routes(self) -> Any:
        bp = self.blueprint

        @bp.route(f"{self.APP_PREFIX}/", defaults={"path": ""})
        def serve_react_app(path):
            if path == "":
                # Server-side permanent redirect to canonical /mv
                return redirect(f"{self.APP_PREFIX}/mv", code=301)
            
            fullpath = os.path.join(self.REACT_BUILD_PATH, path)
            if not os.path.exists(fullpath):
                self.logger.info(f"fullpath {fullpath} does not exist, serving index.html")
                return send_from_directory(self.REACT_BUILD_PATH, "index.html")
            return send_from_directory(self.REACT_BUILD_PATH, path)
        
        @bp.route(f"{self.APP_PREFIX}/<path:path>")
        def serve_react_app_path(path):
            fullpath = os.path.join(self.REACT_BUILD_PATH, path)
            if not os.path.exists(fullpath):
                self.logger.info(f"fullpath {fullpath} does not exist, serving index.html")
                return send_from_directory(self.REACT_BUILD_PATH, "index.html")
            return send_from_directory(self.REACT_BUILD_PATH, path)

        #@bp.route("/run", methods=["POST"])
        @bp.route(f"{self.APP_PREFIX}/run", methods=["POST"])
        def run_mv():
            data = request.json or request.form
            self.utilities_service.log_user_activity(data)
            etfl = data.get("etflist", "").split(",") if data.get("etflist") else ["VOO","VXUS","AVUV","AVDV","AVEM"]
            short  = int(data.get("short", 0))
            maxuse = int(data.get("maxuse", 0))
            normal = int(data.get("normal", 1))
            sd = int(data.get("startdate", 199302))
            ed = int(data.get("enddate",   202312))

            result = mv(
                # @SDS begin
                # note: .copy() is a shallow copy
                #global_data.copy(), 
                self.data_service.get_global_data().copy(),
                # @SDS end
                etfl,
                short, 
                maxuse, 
                normal,
                sd, 
                ed
            )

            #log.info("result: " + str(result))
            return jsonify(result)

    ##
    ## Cannot see where this may be used?
    ##
    # def extract_ols_summary(self, model):
    #     """Extract and structure OLS summary data in a JSON-serializable format."""

    #         # Regression residuals
    #     resid = model.resid

    #     # Durbin-Watson statistic
    #     dw_stat = durbin_watson(resid)

    #     # Jarque-Bera test: returns JB statistic, p-value, skewness, kurtosis
    #     jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(resid)

    #     # Other common metrics
    #     r2 = model.rsquared
    #     r2_adj = model.rsquared_adj
    #     n_obs = int(model.nobs)
    #     annualized_alpha = float(model.params[0]) * 12  # constant term * 12

    #     # Aggregate into dict
    #     diagnostics = {
    #         "r_squared": round(r2, 4),
    #         "adj_r_squared": round(r2_adj, 4),
    #         "n_observations": n_obs,
    #         "alpha_annualized": round(annualized_alpha, 4),
    #         "durbin_watson": round(dw_stat, 4),
    #         "jarque_bera_stat": round(jb_stat, 4),
    #         "jarque_bera_pval": round(jb_pval, 6),
    #         "skewness": round(jb_skew, 4),
    #         "kurtosis": round(jb_kurt, 4),
    #     }


    #     summary = {
    #         "r_squared": round(model.rsquared, 4),
    #         "adj_r_squared": round(model.rsquared_adj, 4),
    #         "f_statistic": round(model.fvalue, 4) if model.fvalue is not None else None,
    #         "prob_f_stat": round(model.f_pvalue, 4) if model.f_pvalue is not None else None,
    #         "n_obs": int(model.nobs),
    #         "aic": round(model.aic, 4),
    #         "bic": round(model.bic, 4),
    #         "df_resid": int(model.df_resid),
    #         "df_model": int(model.df_model),
    #         "log_likelihood": round(model.llf, 4),
    #         "cov_type": model.cov_type,
    #         "coefficients": [],
    #         "diagnostics": diagnostics
    #     }

    #     conf_int_df = model.conf_int()

    #     for i, name in enumerate(model.model.exog_names):
    #         summary["coefficients"].append({
    #             "factor": name,
    #             "coef": round(model.params[i], 4),
    #             "std_err": round(model.bse[i], 4),
    #             "t": round(model.tvalues[i], 4),
    #             "p_value": round(model.pvalues[i], 4),
    #             "ci_lower": round(conf_int_df.iloc[i, 0], 4),
    #             "ci_upper": round(conf_int_df.iloc[i, 1], 4)
    #         })

    #     return summary
