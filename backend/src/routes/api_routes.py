import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import yfinance as yf
import statsmodels.api as sm
from contextlib import redirect_stdout
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from datetime import datetime
from typing import Any
from flask import Blueprint, jsonify, request, g, send_from_directory, make_response, redirect
from mv import mv
from src.logging.app_logger import AppLogger
from src.config.config import Config
from src.services.data_service import DataService
from src.services.file_service import FileService
from src.services.backtest_service import BacktestService
from src.services.backtest_input_error import BacktestInputError
from life_cycle import (
    LifeCycleInputError,
    load_vector_from_csv,
    run_life_cycle_analysis,
)

class ApiRoutes(object):

    def __init__(self) -> None:
        self.logger = AppLogger.get_logger()
        self.logger.info("This is AppRoutes constructor")

        self.APP_PREFIX = os.getenv("APP_PREFIX", "")  # "/financial_analyzer" or ""
        self.logger.info("self.APP_PREFIX: " + str(self.APP_PREFIX))

        routes_dir = os.path.dirname(__file__)
        #self.logger.info("routes_dir: " + str(routes_dir))
        backend_root = os.path.abspath(os.path.join(routes_dir, "..", ".."))  # up to backend
        #self.logger.info("backend_root: " + str(backend_root))
        self.REACT_BUILD_PATH = os.path.join(backend_root, Config.get_property("react.build.dir"))
        #self.logger.info("REACT_BUILD_PATH: " + str(self.REACT_BUILD_PATH))

        self.blueprint = Blueprint("ApiRoutes", __name__)
        self._add_routes()

        self.data_service = DataService()
        self.file_service = FileService()
        # self.matrix_service = MatrixService()
        self.backtest_service = BacktestService()

    def _add_routes(self) -> Any:
        bp = self.blueprint

            # Redirect logic for root URL
        @bp.route("/")
        def redirect_to_app_prefix():
            self.logger.info("ROUTE /")
            self.logger.info("APP_PREFIX is " + str(self.APP_PREFIX))
            # If APP_PREFIX is "/", no need to redirect
            if self.APP_PREFIX == "/" or self.APP_PREFIX == "":
                return self.serve_react_app("")
            # Otherwise redirect to APP_PREFIX with trailing slash
            return redirect(self.APP_PREFIX + "/", code=301)

        @bp.route(f"{self.APP_PREFIX}/", defaults={"path": ""})
        @bp.route(f"{self.APP_PREFIX}/<path:path>")
        def serve_react_app(path):
            self.logger.info("self.APP_PREFIX: " + str(self.APP_PREFIX))
            self.logger.info("path: " + str(path))
            #react_build = current_app.config.get("REACT_BUILD_PATH", "react_build")
            #fullpath = os.path.join(react_build, path)
            fullpath = os.path.join(self.REACT_BUILD_PATH, path)
            # if not os.path.exists(fullpath):
            #     self.logger.info("fullpath: " + str(fullpath) + " DOES NOT EXIST")
            # else:
            #     self.logger.info("fullpath: " + str(fullpath) + " exists")
            #self.logger.info("fullpath: " + str(fullpath))

            if path == "":
                #self.logger.info("sending index.html from: " + str(self.REACT_BUILD_PATH) + "/index.html because path is empty")
                return send_from_directory(self.REACT_BUILD_PATH, "index.html")

            if not os.path.exists(fullpath):
                self.logger.info("fullpath: " + str(fullpath) + " does not exist")
                self.logger.info("sending index.html from: " + str(self.REACT_BUILD_PATH) + "/index.html because fullpath does not exist")
                return send_from_directory(self.REACT_BUILD_PATH, "index.html")
            
            #self.logger.info("sending: " + str(self.REACT_BUILD_PATH) + " " + str(path))
            return send_from_directory(self.REACT_BUILD_PATH, path)

        @bp.route("/run", methods=["POST"])
        def run_mv():
            data = request.json or request.form
            self.log_user_activity(data)
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
    
        @bp.route("/life-cycle/run", methods=["POST"])
        def run_life_cycle():
            self.log_user_activity()

            try:
                returns_file = request.files.get("returns_file")
                cashflows_file = request.files.get("cashflows_file")

                returns_vector = load_vector_from_csv(returns_file, "Return")
                cashflow_vector = load_vector_from_csv(cashflows_file, "Cash flow")

                form_data = request.form or {}
                if not form_data:
                    form_data = request.json or {}

                initial_wealth = self._parse_float(form_data.get("initial_wealth", 0), "Initial wealth", 0.0)
                wmin_cutoff = self._parse_float(form_data.get("wmin_cutoff", 0), "Minimum wealth cutoff", 0.0)
                nsim = self._parse_int(form_data.get("nsim", 1000), "Number of simulations", 1000)

                result = run_life_cycle_analysis(
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

        @bp.route("/backtest", methods=["POST"])
        def run_backtest():
            try:
                data = request.json
                self.log_user_activity(data)

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

        @bp.route("/regression", methods=["POST"])
        def run_regression():
            try:
                # @SDS begin
                #global final_data # Ensure final_data is accessible if it's a global variable
                final_data = self.data_service.get_final_data()
                # @SDS end

                data = request.json
                self.log_user_activity(data)
                ticker = data.get("ticker")
                start_date_str = data.get("start_date", "1970-01-01")
                end_date_str = data.get("end_date", "2023-12-31")

                def parse_date_string(raw_value, label):
                    try:
                        clean_value = (raw_value or "").replace("-", "")
                        if len(clean_value) < 6:
                            raise ValueError
                        return int(clean_value[:6])
                    except (AttributeError, ValueError, TypeError):
                        raise RegressionInputError(f"{label} is invalid. Please use YYYY-MM-DD format.")

                start_date = parse_date_string(start_date_str, "Start date")
                end_date = parse_date_string(end_date_str, "End date")

                model_name_req = data.get("model", "CAPM")

                rolling_period_raw = data.get("rolling_period", 36)
                try:
                    rolling_period = int(rolling_period_raw)
                except (ValueError, TypeError):
                    raise RegressionInputError("Rolling period must be an integer number of months.")
                if rolling_period <= 0:
                    raise RegressionInputError("Rolling period must be a positive integer.")
                
                if not ticker or not str(ticker).strip():
                    raise RegressionInputError("Ticker not provided")

                ticker = str(ticker).strip().upper()

                data_short = final_data[final_data["ticker_new"] == ticker].copy()

                if data_short.empty:
                    raise RegressionInputError(f"No data found for ticker: {ticker}")

                info_messages = []
                warning_messages = []
                error_messages = []

                max_available_date = data_short["date"].max()
                min_available_date = data_short["date"].min()

                def format_ym(value):
                    value_str = f"{int(value):06d}"
                    return f"{value_str[:4]}-{value_str[4:6]}"

                if end_date > max_available_date:
                    info_messages.append(
                        f"End date adjusted to {format_ym(max_available_date)} because that is the latest available data for {ticker}."
                    )
                    end_date = max_available_date
                if start_date < min_available_date:
                    info_messages.append(
                        f"Start date adjusted to {format_ym(min_available_date)} because that is the first available data for {ticker}."
                    )
                    start_date = min_available_date

                if start_date > end_date:
                    raise RegressionInputError(
                        "Start date cannot be after end date for the selected ticker's available range."
                    )

                data_short = data_short[(data_short["date"] >= start_date) & (data_short["date"] <= end_date)]

                if data_short.empty:
                    raise RegressionInputError(
                        f"No data for ticker {ticker} in the specified date range {format_ym(start_date)} - {format_ym(end_date)}"
                    )

                # Prepare y_var (dependent variable: excess returns)
                # Ensure 'RF' (risk-free rate) is present and numeric
                if 'RF' not in data_short.columns:
                    return jsonify({"error": "RF (Risk-Free rate) column missing in data_short."}), 500
                data_short['RF'] = pd.to_numeric(data_short['RF'], errors='coerce')
                data_short['ret'] = pd.to_numeric(data_short['ret'], errors='coerce')

                y_var_series = (data_short["ret"] - data_short["RF"]).rename('y_excess_return') # Rename for clarity

                nobs_initial = y_var_series.shape[0]
                if nobs_initial == 0:
                    raise RegressionInputError("No observations found for the selected ticker and date range.")

                # Select factors based on model_name_req
                factor_columns_map = {
                    "CAPM": ["Mkt-RF"],
                    "FF3": ["Mkt-RF", "HML", "SMB"],
                    "FF4": ["Mkt-RF", "HML", "SMB", "MOM"],
                    "FF5": ["Mkt-RF", "HML", "SMB", "CMA", "RMW"]
                }
                if model_name_req not in factor_columns_map:
                    raise RegressionInputError("Invalid model selected")

                factor_names = factor_columns_map[model_name_req]

                # Ensure factor columns exist and are numeric
                for factor in factor_names:
                    if factor not in data_short.columns:
                        return jsonify({"error": f"Factor column '{factor}' missing in data_short."}), 500
                    data_short[factor] = pd.to_numeric(data_short[factor], errors='coerce')

                x_var_df_factors_only = data_short[factor_names].copy()

                # Add constant and align data by dropping NaNs from the combined DataFrame
                x_var_with_constant_df = sm.add_constant(x_var_df_factors_only, has_constant='add', prepend=True)

                # Align y_var with x_var_with_constant_df using their common index from data_short
                # The index of data_short is used by y_var_series and x_var_df_factors_only
                combined_for_regression = pd.concat([y_var_series, x_var_with_constant_df], axis=1)
                combined_for_regression.dropna(inplace=True) # Drop rows with NaNs in y or any x

                min_required_obs = x_var_with_constant_df.shape[1] + 1
                if combined_for_regression.shape[0] < min_required_obs:
                    raise RegressionInputError(
                        "Not enough data points for regression after handling missing values."
                    )

                y_var_final = combined_for_regression['y_excess_return']
                x_var_final_with_const = combined_for_regression.drop(columns=['y_excess_return'])

                # Ensure column order for exog_names matches params order (sm.OLS should handle this if df passed)
                mdl = sm.OLS(y_var_final, x_var_final_with_const).fit()

                regression_text_html = mdl.summary().as_html()

                # Calculate Return Contribution
                return_contribution_list = []
                # Overall excess return for the ticker
                avg_ann_ticker_excess_ret = np.nanmean(y_var_final) * 12
                return_contribution_list.append({
                    "Factor": ticker,
                    "Av. Ann. Excess Return": avg_ann_ticker_excess_ret,
                    "Return Contribution": 100.0 # By definition for itself
                })

                # Alpha contribution
                alpha_coeff = mdl.params.get('const', 0.0) # Default to 0 if 'const' not found
                alpha_annualized_val = alpha_coeff * 12
                alpha_contribution_pct = (alpha_annualized_val / avg_ann_ticker_excess_ret * 100) if avg_ann_ticker_excess_ret else None
                return_contribution_list.append({
                    "Factor": "alpha",
                    "Av. Ann. Excess Return": alpha_annualized_val,
                    "Return Contribution": alpha_contribution_pct
                })

                # Factor contributions
                # x_var_df_factors_only_aligned ensures means are from the same sample used in regression
                x_var_df_factors_only_aligned = x_var_df_factors_only.loc[y_var_final.index]

                for factor_name in factor_names:
                    if factor_name in mdl.params.index: # Check if factor was included (not dropped due to collinearity etc.)
                        factor_loading = mdl.params[factor_name]
                        avg_factor_return = np.nanmean(x_var_df_factors_only_aligned[factor_name]) * 12
                        contribution_value = factor_loading * avg_factor_return
                        contribution_pct = (contribution_value / avg_ann_ticker_excess_ret * 100) if avg_ann_ticker_excess_ret else None
                        return_contribution_list.append({
                            "Factor": factor_name,
                            "Av. Ann. Excess Return": avg_factor_return,
                            "Return Contribution": contribution_pct
                        })


                # Rolling Regression Plot Data
                image_urls = [] # Keep this empty if the frontend fully handles plotting
                rolling_plot_data_for_json = None

                nobs_final = len(y_var_final) # Number of observations after NaN handling for overall regression
                if nobs_final >= rolling_period + 10:
                    # Ensure rolling regression uses the same cleaned/aligned data
                    out_roll = np.full((nobs_final - rolling_period + 1, x_var_final_with_const.shape[1]), np.nan)

                    # Align data_short for date indexing with the cleaned data for rolling period
                    aligned_dates_for_rolling_idx = y_var_final.index
                    date_series_for_rolling = data_short.loc[aligned_dates_for_rolling_idx, "date"]

                    for k_loop_idx in range(rolling_period -1, nobs_final): # Iterate using index positions
                        start_idx_rolling = k_loop_idx - (rolling_period -1)
                        end_idx_rolling = k_loop_idx + 1

                        y_roll_s = y_var_final.iloc[start_idx_rolling:end_idx_rolling]
                        x_roll_df = x_var_final_with_const.iloc[start_idx_rolling:end_idx_rolling]

                        if not x_roll_df.empty and not y_roll_s.empty and len(y_roll_s) >= x_var_final_with_const.shape[1]:
                            try:
                                mdl_roll = sm.OLS(y_roll_s, x_roll_df, missing='drop').fit()
                                out_roll[k_loop_idx - (rolling_period - 1), :] = mdl_roll.params.values
                            except Exception as e_roll:
                                window_end_raw = None
                                try:
                                    window_end_raw = date_series_for_rolling.iloc[k_loop_idx]
                                except Exception:
                                    window_end_raw = None

                                window_end_fmt = (
                                    format_ym(window_end_raw)
                                    if window_end_raw is not None and not pd.isna(window_end_raw)
                                    else f"index {k_loop_idx}"
                                )

                                error_message = (
                                    f"Rolling regression failed for the window ending {window_end_fmt}: {e_roll}"
                                )
                                error_messages.append(error_message)
                                print(error_message)
                                out_roll[k_loop_idx - (rolling_period - 1), :] = np.nan
                        else:
                            out_roll[k_loop_idx - (rolling_period - 1), :] = np.nan

                    # Dates for the rolling plot (end of each window)
                    date_aux_series_rolling = date_series_for_rolling.iloc[rolling_period - 1:].astype(str)
                    dates_aux_dt_rolling = pd.to_datetime(date_aux_series_rolling, format="%Y%m")
                    plot_dates_str = [date_obj.strftime('%Y-%m-%d') for date_obj in dates_aux_dt_rolling]

                    # Alpha series (constant term is the first column in x_var_final_with_const)
                    alpha_values_rolling = (out_roll[:, 0] * 12).tolist() # Assuming 'const' is always first

                    # Factor loadings series
                    factor_loadings_series_list = []
                    # exog_names_rolling should be consistent, taken from x_var_final_with_const.columns
                    # factor_names are ['Mkt-RF', 'HML', ...]
                    # x_var_final_with_const.columns are ['const', 'Mkt-RF', 'HML', ...]
                    for i, factor_name_plot in enumerate(factor_names):
                        # Find the index of this factor in the full exog list (including const)
                        if factor_name_plot in x_var_final_with_const.columns:
                            factor_col_idx_in_out_roll = x_var_final_with_const.columns.get_loc(factor_name_plot)
                            factor_loadings_series_list.append({
                                "name": factor_name_plot,
                                "values": out_roll[:, factor_col_idx_in_out_roll].tolist()
                            })

                    rolling_plot_data_for_json = {
                        "dates": plot_dates_str,
                        "alpha_series": alpha_values_rolling,
                        "factor_series": factor_loadings_series_list,
                        "factor_names": factor_names # Original factor names for legend
                    }
                else:
                    warning_messages.append(
                        "Not enough observations to generate the rolling regression chart."
                    )

                info_messages.append(
                    f"Regression run for {ticker} from {format_ym(start_date)} to {format_ym(end_date)} using the {model_name_req} model."
                )

                response_payload = {
                    "summary_table": return_contribution_list, # Use the list of dicts
                    "image_urls": image_urls,
                    "regression_output": {
                        "r_squared": round(mdl.rsquared, 4),
                        "adj_r_squared": round(mdl.rsquared_adj, 4),
                        "alpha_annualized": round(alpha_annualized_val, 4), # Use the correctly calculated alpha
                        "n_observations": int(mdl.nobs),
                        "text_summary": regression_text_html
                    },
                    "rolling_plot_data": rolling_plot_data_for_json,
                    "messages": info_messages,
                    "warnings": warning_messages,
                    "errors": error_messages
                }

                def sanitize_for_json(data_to_sanitize): # Renamed variable to avoid conflict
                    if isinstance(data_to_sanitize, dict):
                        return {k: sanitize_for_json(v) for k, v in data_to_sanitize.items()}
                    elif isinstance(data_to_sanitize, list):
                        return [sanitize_for_json(i) for i in data_to_sanitize]
                    elif isinstance(data_to_sanitize, (np.float64, np.float32, float)):
                        return None if np.isnan(data_to_sanitize) else float(data_to_sanitize)
                    elif isinstance(data_to_sanitize, (np.int64, np.int32, np.int_, int)):
                        return int(data_to_sanitize)
                    elif isinstance(data_to_sanitize, (np.bool_, np.bool, bool)):
                        return bool(data_to_sanitize)
                    elif pd.isna(data_to_sanitize):
                        return None
                    return data_to_sanitize

                return jsonify(sanitize_for_json(response_payload))

            except RegressionInputError as e:
                return jsonify({"error": str(e), "errors": getattr(e, "errors", [str(e)])}), 400
            except Exception as e:
                import traceback
                current_traceback = traceback.format_exc() # Capture traceback string
                print(current_traceback) # Print to server logs
                return jsonify({"error": str(e), "trace": current_traceback}), 500

        # this is for user uploads/downloads
        @bp.route('/static/<path:filename>')
        def serve_image(filename):
            self.log_user_activity()
            # @SDS begin
            # return send_from_directory(STATIC_DIR, filename)
            return send_from_directory(self.file_service.get_STATIC_DIR(), filename)
            # @SDS end

    def convert_numpy(self, obj):
        """Helper function to convert numpy types to native Python."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

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

class RegressionInputError(Exception):
    """Raised when regression input fails validation."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or [message]