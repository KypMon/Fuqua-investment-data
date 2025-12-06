import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from flask import Blueprint, request, jsonify, g
from datetime import datetime
from src.middleware.fw_user import FwUser
from src.services.backtest_service import BacktestService
from src.services.backtest_input_error import BacktestInputError
from src.services.file_service import FileService
from src.services.utilities_service import UtilitiesService
from src.logging.app_logger import AppLogger

class Backtest(object):

    def __init__(self, app=None) -> None:
        self.logger = AppLogger.get_logger()
        self.utilities_service = UtilitiesService()
        self.file_service = FileService()
        self.backtest_service = BacktestService()

        self.APP_PREFIX = os.getenv("APP_PREFIX", "")  # "/financial_analyzer" or ""
        self.logger.info("APP_PREFIX: " + self.APP_PREFIX)

        self.blueprint = Blueprint(
            "Backtest",
            __name__,
            url_prefix=f"{self.APP_PREFIX}/backtest",
            #static_url_path="/static",     # served at /backtest/static
            #static_folder=static_dir,
        )

        self.blueprint.add_url_rule(
            "/run",
            view_func=self.run_backtest,
            methods=["POST"],
        )

        if app:
            app.register_blueprint(self.blueprint)

    #@bp.route(f"{self.APP_PREFIX}/backtest", methods=["POST"])
    def run_backtest(self):
        try:
            data = request.json
            self.utilities_service.log_user_activity(data)

            start_date_str = data.get("start_date", "1970-01-01")
            end_date_str = data.get("end_date", "2023-12-31")
            start_date = int(start_date_str.replace("-", "")[:6])
            end_date = int(end_date_str.replace("-", "")[:6])
            
            tickers = data.get("tickers", [])
            
            allocation1_raw = data.get("allocation1", [])
            allocation2_raw = data.get("allocation2", [])
            allocation3_raw = data.get("allocation3", [])

            num_tickers = len(tickers)
            
            def process_allocation(raw_alloc, length):
                processed = [np.nan] * length
                for i, x_val_str in enumerate(raw_alloc):
                    if i < length:
                        if x_val_str is not None and x_val_str != "":
                            try:
                                processed[i] = float(x_val_str)
                            except ValueError:
                                processed[i] = np.nan
                        else:
                            processed[i] = np.nan
                return processed

            allocation1 = np.array(process_allocation(allocation1_raw, num_tickers), dtype=float)
            allocation2 = np.array(process_allocation(allocation2_raw, num_tickers), dtype=float)
            allocation3 = np.array(process_allocation(allocation3_raw, num_tickers), dtype=float)
            
            rebalancing = data.get("rebalance", "monthly")
            benchmark_input = data.get("benchmark", ["CRSPVW"])
            benchmark = benchmark_input[0] if isinstance(benchmark_input, list) and benchmark_input else "CRSPVW"
            start_balance = float(data.get("start_balance", 10000))

            f = io.StringIO()
            plt.switch_backend("Agg")
            
            structured_results_from_backtesting = {} 

            with redirect_stdout(f):
                structured_results_from_backtesting = self.backtest_service.backtesting(
                    start_date, end_date, tickers,
                    allocation1, allocation2, allocation3,
                    rebalancing, benchmark, start_balance
                )
            
            output_text = f.getvalue()

            image_urls = []
            timestamp = datetime.now().timestamp()
            # for i, fig_num in enumerate(plt.get_fignums()): # If plt.show() was indeed removed, this loop might not find figures.
            #     fig = plt.figure(fig_num)
            #     img_filename = f"backtest_plot_{timestamp}_{i}.png"
            #     path = os.path.join(STATIC_DIR, img_filename)
            #     fig.savefig(path)
            #     image_urls.append(f"/static/{img_filename}")
            # plt.close("all")

            response_data = {
                "output_text": output_text,
                "image_urls": image_urls, # Can be empty if all plots are now frontend-rendered
                "portfolio_allocations": structured_results_from_backtesting.get("portfolio_allocations", []),
                "summary_table": structured_results_from_backtesting.get("performance_summary_table", []),
                "drawdown_tables": structured_results_from_backtesting.get("drawdown_tables", []),
                "regression_table": structured_results_from_backtesting.get("regression_summary_tables", []),
                "portfolio_growth_plot_data": structured_results_from_backtesting.get("portfolio_growth_plot_data", []),
                "annual_returns_plot_data": structured_results_from_backtesting.get("annual_returns_plot_data", {}),
                "drawdown_plot_data": structured_results_from_backtesting.get("drawdown_plot_data", []),
                "messages": structured_results_from_backtesting.get("messages", []),
                "warnings": structured_results_from_backtesting.get("warnings", [])
            }
            
            def sanitize_for_json(data):
                if isinstance(data, dict):
                    return {k: sanitize_for_json(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [sanitize_for_json(i) for i in data]
                elif isinstance(data, (np.float64, np.float32, float)): # Added float here
                    return None if np.isnan(data) else float(data)
                elif isinstance(data, (np.int64, np.int32, np.int_, int)): # Added int here
                    return int(data)
                elif isinstance(data, (np.bool_, bool)): # Added bool here
                    return bool(data)
                elif pd.isna(data):
                    return None
                return data

            sanitized_response_data = sanitize_for_json(response_data)
            return jsonify(sanitized_response_data)

        except BacktestInputError as e:
            return jsonify({"error": str(e), "errors": getattr(e, "errors", [str(e)])}), 400
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


    def get_blueprint(self):
        return Backtest.blueprint