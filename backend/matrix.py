"""Utilities for the custom mean-variance module ("matrix")."""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from cvxopt import matrix, solvers

# Disable solver output globally for cleaner logs
solvers.options["show_progress"] = False


@dataclass
class MatrixPayload:
    columns: List[str]
    records: List[Dict[str, object]]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "MatrixPayload":
        return cls(columns=list(df.columns), records=df.to_dict(orient="records"))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.columns:
            raise ValueError("No columns provided in payload")
        if not isinstance(self.records, list):
            raise ValueError("Matrix payload expects a list of records")
        return pd.DataFrame(self.records, columns=self.columns)


def parse_matrix_payload(payload: Optional[dict]) -> pd.DataFrame:
    """Convert a JSON payload into a :class:`pandas.DataFrame`."""
    if payload is None:
        raise ValueError("Matrix payload is missing")

    if isinstance(payload, MatrixPayload):
        return payload.to_dataframe()

    if "records" in payload:
        records = payload["records"]
        columns = payload.get("columns")
        if columns is None and records:
            columns = list(records[0].keys())
        df = pd.DataFrame(records, columns=columns)
        return df

    if "rows" in payload and "columns" in payload:
        return pd.DataFrame(payload["rows"], columns=payload["columns"])

    raise ValueError("Unsupported matrix payload format")


def _clean_date_column(df: pd.DataFrame, column: str = "Date") -> pd.DataFrame:
    if column not in df.columns:
        return df

    cleaned = df.copy()
    cleaned[column] = (
        cleaned[column]
        .apply(lambda x: int(str(x).strip()) if str(x).strip() else np.nan)
        .astype("Int64")
    )
    cleaned[column] = cleaned[column].astype("Int64")
    return cleaned


def download_matret(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Download monthly returns from Yahoo! Finance.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols. Empty strings are ignored.
    start_date, end_date:
        ISO formatted date strings (``YYYY-MM-DD``).

    Returns
    -------
    DataFrame and list
        A tuple with the ``matret`` dataframe and the list of tickers that
        successfully returned data.
    """

    normalized_tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not normalized_tickers:
        raise ValueError("At least one ticker is required")

    raw = yf.download(
        tickers=normalized_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError("No price data returned for the requested tickers")

    print(raw)

    adj_close = raw["Close"] if "Close" in raw else raw
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()

    adj_close = adj_close.ffill().dropna(how="all")
    available = [col for col in adj_close.columns if col in normalized_tickers]
    if not available:
        raise ValueError("No usable price series were returned")

    prices = adj_close[available]

    # Daily simple returns
    daily_returns = prices.pct_change().dropna(how="all")
    if daily_returns.empty:
        raise ValueError("Insufficient data to compute returns")

    # Convert to monthly discrete returns using compounded daily returns
    monthly_returns = (1 + daily_returns).resample("M").prod() - 1
    monthly_returns = monthly_returns.dropna(how="all")

    if monthly_returns.empty:
        raise ValueError("Monthly returns could not be computed")

    date_codes = monthly_returns.index.to_period("M").strftime("%Y%m")
    matret = monthly_returns.reset_index(drop=True)
    matret.insert(0, "Date", date_codes.astype(int))

    matret = matret.dropna(axis=1, how="all")
    matret = _clean_date_column(matret, "Date")

    return matret, available


def create_mat_er_covr(matret: pd.DataFrame, risk_free: Optional[float] = None) -> Tuple[pd.DataFrame, Optional[float]]:
    """Compute the mean/covariance matrix from ``matret``."""
    if matret is None or matret.empty:
        raise ValueError("matret is empty")

    df = matret.copy()
    df.columns = [str(col).strip() for col in df.columns]

    date_columns = {col for col in df.columns if col.lower() in {"date", "datetime"}}
    for col in date_columns:
        df.drop(columns=col, inplace=True)

    rf_column = None
    for col in df.columns:
        if str(col).strip().lower() in {"rf", "riskfree", "risk_free"}:
            rf_column = col
            break

    rf_value = None
    if rf_column is not None:
        rf_series = pd.to_numeric(df[rf_column], errors="coerce").dropna()
        if not rf_series.empty:
            rf_value = float(rf_series.mean())
        df.drop(columns=rf_column, inplace=True)

    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if numeric.empty:
        raise ValueError("No valid numeric data found in matret")

    means = numeric.mean()
    cov = numeric.cov()

    assets = list(numeric.columns)

    ordered_means = means.loc[assets]
    ordered_cov = cov.loc[assets, assets]

    matrix = ordered_cov.copy()
    matrix.insert(0, "Mean", ordered_means.values)
    matrix.index.name = "Assets"
    result = matrix.reset_index()

    resolved_rf = risk_free if risk_free is not None else rf_value
    if resolved_rf is not None:
        rf_row = {"Assets": "rf", "Mean": float(resolved_rf)}
        for asset in assets:
            rf_row[asset] = np.nan
        result = pd.concat([result, pd.DataFrame([rf_row])], ignore_index=True)

    return result, resolved_rf


def _build_covariance(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[float]]:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # Locate the column that represents the expected return vector.
    mean_column: Optional[str] = None
    for candidate in df.columns:
        normalized = str(candidate).strip().lower()
        if normalized in {"mean", "er", "expected_return", "expectedreturn"}:
            mean_column = candidate
            break
    if mean_column is None:
        raise ValueError("Could not identify the mean return column in mat_er_covr")

    # Locate a column that might contain asset labels (optional).
    asset_column: Optional[str] = None
    for candidate in df.columns:
        if candidate == mean_column:
            continue
        normalized = str(candidate).strip().lower()
        if normalized in {"assets", "asset", "ticker", "tickers"}:
            asset_column = candidate
            break
        if normalized.startswith("unnamed"):
            column_values = df[candidate].astype(str).str.strip()
            if column_values.replace("", np.nan).notna().any():
                asset_column = candidate
                break

    rf_value: Optional[float] = None
    assets: Optional[List[str]] = None

    if asset_column is not None:
        asset_series = df[asset_column].astype(str).str.strip()
        rf_mask = asset_series.str.lower() == "rf"
        if rf_mask.any():
            rf_series = pd.to_numeric(df.loc[rf_mask, mean_column], errors="coerce").dropna()
            if not rf_series.empty:
                rf_value = float(rf_series.iloc[0])
            df = df.loc[~rf_mask].copy()
            asset_series = asset_series.loc[~rf_mask]
        assets = asset_series.tolist()
        df.drop(columns=[asset_column], inplace=True)

    if df.empty:
        raise ValueError("No asset rows found in mat_er_covr")

    means = pd.to_numeric(df[mean_column], errors="coerce")
    if means.isna().any():
        raise ValueError("Mean column contains non-numeric values")

    # Candidate covariance columns are everything except the mean column.
    cov_candidate_cols = [col for col in df.columns if col != mean_column]
    if not cov_candidate_cols:
        raise ValueError("Covariance matrix columns not found in mat_er_covr")

    cov_df = df[cov_candidate_cols].apply(pd.to_numeric, errors="coerce")

    if assets is None:
        # Attempt to infer and remove a risk-free row by looking for NaNs across the covariance block.
        rf_rows = cov_df.isna().all(axis=1)
        if rf_rows.any():
            rf_series = means[rf_rows].dropna()
            if not rf_series.empty:
                rf_value = float(rf_series.iloc[0])
            df = df.loc[~rf_rows].copy()
            means = means.loc[~rf_rows]
            cov_df = cov_df.loc[~rf_rows]

    if df.empty:
        raise ValueError("No asset rows found in mat_er_covr")

    if cov_df.isna().any().any():
        raise ValueError("Covariance matrix contains invalid values")

    if assets is not None:
        # Re-order covariance columns to align with the asset order.
        normalized_columns = {str(col).strip().lower(): col for col in cov_df.columns}
        ordered_columns: List[str] = []
        for asset in assets:
            key = str(asset).strip().lower()
            variants = [key, f"cov_{key}"]
            matched: Optional[str] = None
            for variant in variants:
                if variant in normalized_columns:
                    matched = normalized_columns[variant]
                    break
            if matched is None:
                raise ValueError(f"Covariance columns missing for asset '{asset}'")
            ordered_columns.append(matched)
        cov_df = cov_df[ordered_columns]
    else:
        inferred_assets: List[str] = []
        for col in cov_df.columns:
            name = str(col).strip()
            lower = name.lower()
            if lower.startswith("cov_"):
                suffix = name.split("_", 1)[1]
                inferred_assets.append(suffix if suffix else name)
            else:
                inferred_assets.append(name)
        assets = inferred_assets

    if cov_df.shape[0] != len(assets) or cov_df.shape[1] != len(assets):
        raise ValueError("Covariance matrix dimensions do not match asset count")

    cov_matrix = cov_df.to_numpy()
    return means.to_numpy(), cov_matrix, assets, rf_value


def _solve_min_variance(cov: np.ndarray, allow_short: bool) -> np.ndarray:
    n = cov.shape[0]
    P = matrix(cov)
    q = matrix(np.zeros(n))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    if allow_short:
        solution = solvers.qp(P, q, None, None, A, b)
    else:
        G = matrix(-np.eye(n))
        h = matrix(0.0, (n, 1))
        solution = solvers.qp(P, q, G, h, A, b)
    weights = np.array(solution["x"]).flatten()
    return weights


def _solve_target_return(
    cov: np.ndarray,
    means: np.ndarray,
    target: float,
    allow_short: bool,
) -> Optional[np.ndarray]:
    n = cov.shape[0]
    P = matrix(cov)
    q = matrix(np.zeros(n))
    A = matrix(np.vstack((np.ones(n), means)))
    b = matrix([1.0, target])
    try:
        if allow_short:
            sol = solvers.qp(P, q, None, None, A, b)
        else:
            G = matrix(-np.eye(n))
            h = matrix(0.0, (n, 1))
            sol = solvers.qp(P, q, G, h, A, b)
    except Exception:
        return None
    if sol is None or sol["status"] not in {"optimal", "optimal_inaccurate"}:
        return None
    return np.array(sol["x"]).flatten()


def compute_portfolios(
    mat_er_covr: pd.DataFrame,
    risk_free: Optional[float] = None,
    grid_size: int = 80,
) -> Dict[str, object]:
    means, cov, assets, rf_from_matrix = _build_covariance(mat_er_covr)
    resolved_rf = risk_free if risk_free is not None else rf_from_matrix
    if resolved_rf is None:
        resolved_rf = 0.0

    def build_block(allow_short: bool) -> Dict[str, object]:
        min_var_weights = _solve_min_variance(cov, allow_short)
        min_var_ret = float(min_var_weights @ means)
        min_var_std = float(np.sqrt(min_var_weights @ cov @ min_var_weights))

        min_return = float(min(means))
        max_return = float(max(means))
        if np.isclose(min_return, max_return):
            ret_targets = np.array([min_return])
        else:
            ret_targets = np.linspace(min_return, max_return, grid_size)

        frontier_weights: List[np.ndarray] = []
        actual_returns: List[float] = []
        actual_stds: List[float] = []

        for target in ret_targets:
            weights = _solve_target_return(cov, means, float(target), allow_short)
            if weights is None:
                continue
            port_std = float(np.sqrt(weights @ cov @ weights))
            port_ret = float(weights @ means)
            frontier_weights.append(weights)
            actual_returns.append(port_ret)
            actual_stds.append(port_std)

        if not frontier_weights:
            # Fallback: use min variance portfolio only
            frontier_weights = [min_var_weights]
            actual_returns = [min_var_ret]
            actual_stds = [min_var_std]

        sharpe_ratios = []
        for weights, port_ret, port_std in zip(frontier_weights, actual_returns, actual_stds):
            if port_std <= 0:
                sharpe_ratios.append(float("-inf"))
            else:
                sharpe_ratios.append((port_ret - resolved_rf) / port_std)

        max_idx = int(np.argmax(sharpe_ratios)) if sharpe_ratios else 0
        tangency_weights = frontier_weights[max_idx]
        tangency_ret = float(tangency_weights @ means)
        tangency_std = float(np.sqrt(tangency_weights @ cov @ tangency_weights))

        def to_weight_map(weights: np.ndarray) -> Dict[str, float]:
            return {asset: float(weight) for asset, weight in zip(assets, weights)}

        frontier = [
            {"x": float(std), "y": float(ret)}
            for std, ret in zip(actual_stds, actual_returns)
        ]

        return {
            "frontier": frontier,
            "min_variance": {
                "sigma": min_var_std,
                "mean": min_var_ret,
                "weights": to_weight_map(min_var_weights),
            },
            "tangency": {
                "sigma": tangency_std,
                "mean": tangency_ret,
                "weights": to_weight_map(tangency_weights),
                "sharpe": ((tangency_ret - resolved_rf) / tangency_std) if tangency_std > 0 else None,
            },
        }

    return {
        "assets": assets,
        "risk_free": resolved_rf,
        "long_only": build_block(False),
        "short_allowed": build_block(True),
    }


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def timestamped_filename(prefix: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}_{ts}.csv"
