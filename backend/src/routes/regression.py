import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from uuid import uuid4
from flask import Blueprint, request, jsonify, g, url_for, abort, send_file
from datetime import datetime
#from werkzeug.utils import secure_filename
from src.middleware.fw_user import FwUser
from src.services.regression_input_error import RegressionInputError
from src.services.file_service import FileService
from src.services.data_service import DataService
from src.services.utilities_service import UtilitiesService
from src.logging.app_logger import AppLogger

class Regression(object):

    def __init__(self, app=None) -> None:
        self.logger = AppLogger.get_logger()
        self.utilities_service = UtilitiesService()
        self.file_service = FileService()
        self.data_service = DataService()

        self.APP_PREFIX = os.getenv("APP_PREFIX", "")  # "/financial_analyzer" or ""

        # determine absolute path for file uploads/downloads
        server_static_dir = os.getenv("STATIC_DIR")
        if server_static_dir: # server environment
            static_dir = server_static_dir + "/regression" # /static/regression
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
            "Regression",
            __name__,
            url_prefix=f"{self.APP_PREFIX}/regression",
            static_url_path="/static",     # served at /regression/static
            static_folder=static_dir,
        )

        self.blueprint.add_url_rule(
            "/download/html/<token>",
            view_func=self.download_regression_html,
            methods=["GET"],
        )

        self.blueprint.add_url_rule(
            "/download/csv/<token>",
            view_func=self.download_regression_csv,
            methods=["GET"],
        )

        self.blueprint.add_url_rule(
            "/run",
            view_func=self.run_regression,
            methods=["POST"],
        )

        if app:
            app.register_blueprint(self.blueprint)

    #@bp.route(f"{self.APP_PREFIX}/regression/run", methods=["POST"])
    def run_regression(self):
        try:
            final_data = self.data_service.get_final_data()

            data = request.json
            self.utilities_service.log_user_activity(data)
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
            
            # files (multi-user)
            # Save regression outputs for this user 
            user = getattr(g, "fwUser", None)

            # 1) Save the HTML summary
            html_filename = self.file_service.save_html(user, regression_text_html, f"regression_summary_{ticker}", self.static_dir)
            html_path = os.path.join(self.static_dir, html_filename)
            html_url = self.build_download_url_via_token(user, html_path, "Regression.download_regression_html")

            # 2) Save the CSV summary table
            df_summary = pd.DataFrame(return_contribution_list)
            csv_filename = self.file_service.save_dataframe(user, df_summary, f"regression_summary_{ticker}", self.static_dir)
            csv_path = os.path.join(self.static_dir, csv_filename)
            csv_url = self.build_download_url_via_token(user, csv_path, "Regression.download_regression_csv")

            # 3) Add URLs into payload
            response_payload["html_url"] = html_url
            response_payload["csv_url"] = csv_url

            return jsonify(sanitize_for_json(response_payload))

        except RegressionInputError as e:
            return jsonify({"error": str(e), "errors": getattr(e, "errors", [str(e)])}), 400
        except Exception as e:
            import traceback
            current_traceback = traceback.format_exc() # Capture traceback string
            self.logger.error(str(current_traceback)) # Print to server logs
            return jsonify({"error": str(e), "trace": current_traceback}), 500
        
    def download_regression_html(self, token):
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
                mimetype="text/html",
                max_age=0,
                conditional=False
            )
        except FileNotFoundError:
            abort(404)

    def download_regression_csv(self, token):
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

    def build_download_url_via_token(self, fwUser, file_path, urlFor):
        # --- NEW: Generate unguessable token and store mapping ---
        token = uuid4().hex
        self.file_service.register_user_file(fwUser, token, file_path, self.token_dir)

        # Build URL to download via token
        download_url = (url_for(urlFor, token=token)).replace(self.APP_PREFIX, "")
        return download_url
    
    def get_blueprint(self):
        return Regression.blueprint