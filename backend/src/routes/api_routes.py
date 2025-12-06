import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import io
import os
# import statsmodels.api as sm
from contextlib import redirect_stdout
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from datetime import datetime
from typing import Any
from flask import Blueprint, jsonify, request, g, send_from_directory, redirect
from mv import mv
from src.logging.app_logger import AppLogger
from src.config.config import Config
from src.services.data_service import DataService
from src.services.file_service import FileService
from src.services.backtest_service import BacktestService
from src.services.backtest_input_error import BacktestInputError
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

        #@bp.route("/backtest", methods=["POST"])
        # @bp.route(f"{self.APP_PREFIX}/backtest", methods=["POST"])
        # def run_backtest():
        #     try:
        #         data = request.json
        #         self.utilities_service.log_user_activity(data)

        #         start_date_str = data.get("start_date", "1970-01-01")
        #         end_date_str = data.get("end_date", "2023-12-31")
        #         start_date = int(start_date_str.replace("-", "")[:6])
        #         end_date = int(end_date_str.replace("-", "")[:6])
                
        #         tickers = data.get("tickers", [])
                
        #         allocation1_raw = data.get("allocation1", [])
        #         allocation2_raw = data.get("allocation2", [])
        #         allocation3_raw = data.get("allocation3", [])

        #         num_tickers = len(tickers)
                
        #         def process_allocation(raw_alloc, length):
        #             processed = [np.nan] * length
        #             for i, x_val_str in enumerate(raw_alloc):
        #                 if i < length:
        #                     if x_val_str is not None and x_val_str != "":
        #                         try:
        #                             processed[i] = float(x_val_str)
        #                         except ValueError:
        #                             processed[i] = np.nan
        #                     else:
        #                         processed[i] = np.nan
        #             return processed

        #         allocation1 = np.array(process_allocation(allocation1_raw, num_tickers), dtype=float)
        #         allocation2 = np.array(process_allocation(allocation2_raw, num_tickers), dtype=float)
        #         allocation3 = np.array(process_allocation(allocation3_raw, num_tickers), dtype=float)
                
        #         rebalancing = data.get("rebalance", "monthly")
        #         benchmark_input = data.get("benchmark", ["CRSPVW"])
        #         benchmark = benchmark_input[0] if isinstance(benchmark_input, list) and benchmark_input else "CRSPVW"
        #         start_balance = float(data.get("start_balance", 10000))

        #         f = io.StringIO()
        #         plt.switch_backend("Agg")
                
        #         structured_results_from_backtesting = {} 

        #         with redirect_stdout(f):
        #             structured_results_from_backtesting = self.backtest_service.backtesting(
        #                 start_date, end_date, tickers,
        #                 allocation1, allocation2, allocation3,
        #                 rebalancing, benchmark, start_balance
        #             )
                
        #         output_text = f.getvalue()

        #         image_urls = []
        #         timestamp = datetime.now().timestamp()
        #         # for i, fig_num in enumerate(plt.get_fignums()): # If plt.show() was indeed removed, this loop might not find figures.
        #         #     fig = plt.figure(fig_num)
        #         #     img_filename = f"backtest_plot_{timestamp}_{i}.png"
        #         #     path = os.path.join(STATIC_DIR, img_filename)
        #         #     fig.savefig(path)
        #         #     image_urls.append(f"/static/{img_filename}")
        #         # plt.close("all")

        #         response_data = {
        #             "output_text": output_text,
        #             "image_urls": image_urls, # Can be empty if all plots are now frontend-rendered
        #             "portfolio_allocations": structured_results_from_backtesting.get("portfolio_allocations", []),
        #             "summary_table": structured_results_from_backtesting.get("performance_summary_table", []),
        #             "drawdown_tables": structured_results_from_backtesting.get("drawdown_tables", []),
        #             "regression_table": structured_results_from_backtesting.get("regression_summary_tables", []),
        #             "portfolio_growth_plot_data": structured_results_from_backtesting.get("portfolio_growth_plot_data", []),
        #             "annual_returns_plot_data": structured_results_from_backtesting.get("annual_returns_plot_data", {}),
        #             "drawdown_plot_data": structured_results_from_backtesting.get("drawdown_plot_data", []),
        #             "messages": structured_results_from_backtesting.get("messages", []),
        #             "warnings": structured_results_from_backtesting.get("warnings", [])
        #         }
                
        #         def sanitize_for_json(data):
        #             if isinstance(data, dict):
        #                 return {k: sanitize_for_json(v) for k, v in data.items()}
        #             elif isinstance(data, list):
        #                 return [sanitize_for_json(i) for i in data]
        #             elif isinstance(data, (np.float64, np.float32, float)): # Added float here
        #                 return None if np.isnan(data) else float(data)
        #             elif isinstance(data, (np.int64, np.int32, np.int_, int)): # Added int here
        #                 return int(data)
        #             elif isinstance(data, (np.bool_, bool)): # Added bool here
        #                 return bool(data)
        #             elif pd.isna(data):
        #                 return None
        #             return data

        #         sanitized_response_data = sanitize_for_json(response_data)
        #         return jsonify(sanitized_response_data)

        #     except BacktestInputError as e:
        #         return jsonify({"error": str(e), "errors": getattr(e, "errors", [str(e)])}), 400
        #     except Exception as e:
        #         import traceback
        #         traceback.print_exc()
        #         return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    def extract_ols_summary(self, model):
        """Extract and structure OLS summary data in a JSON-serializable format."""

            # Regression residuals
        resid = model.resid

        # Durbin-Watson statistic
        dw_stat = durbin_watson(resid)

        # Jarque-Bera test: returns JB statistic, p-value, skewness, kurtosis
        jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(resid)

        # Other common metrics
        r2 = model.rsquared
        r2_adj = model.rsquared_adj
        n_obs = int(model.nobs)
        annualized_alpha = float(model.params[0]) * 12  # constant term * 12

        # Aggregate into dict
        diagnostics = {
            "r_squared": round(r2, 4),
            "adj_r_squared": round(r2_adj, 4),
            "n_observations": n_obs,
            "alpha_annualized": round(annualized_alpha, 4),
            "durbin_watson": round(dw_stat, 4),
            "jarque_bera_stat": round(jb_stat, 4),
            "jarque_bera_pval": round(jb_pval, 6),
            "skewness": round(jb_skew, 4),
            "kurtosis": round(jb_kurt, 4),
        }


        summary = {
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "f_statistic": round(model.fvalue, 4) if model.fvalue is not None else None,
            "prob_f_stat": round(model.f_pvalue, 4) if model.f_pvalue is not None else None,
            "n_obs": int(model.nobs),
            "aic": round(model.aic, 4),
            "bic": round(model.bic, 4),
            "df_resid": int(model.df_resid),
            "df_model": int(model.df_model),
            "log_likelihood": round(model.llf, 4),
            "cov_type": model.cov_type,
            "coefficients": [],
            "diagnostics": diagnostics
        }

        conf_int_df = model.conf_int()

        for i, name in enumerate(model.model.exog_names):
            summary["coefficients"].append({
                "factor": name,
                "coef": round(model.params[i], 4),
                "std_err": round(model.bse[i], 4),
                "t": round(model.tvalues[i], 4),
                "p_value": round(model.pvalues[i], 4),
                "ci_lower": round(conf_int_df.iloc[i, 0], 4),
                "ci_upper": round(conf_int_df.iloc[i, 1], 4)
            })

        return summary
