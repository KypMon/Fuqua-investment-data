import numpy as np
import pandas as pd
import random

from cvxopt import matrix, solvers


def sharpe_ratio(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float) -> float:
    """Annualized Sharpe ratio for weights ``w``."""
    return (w @ mu - rf) / np.sqrt(w.T @ cov @ w)


def mv(
    df: pd.DataFrame,
    etflist=None,
    short: int = 0,
    maxuse: int = 1,
    normal: int = 1,
    startdate: int = 199302,
    enddate: int = 202312,
):
    """Mean-variance engine returning data for the frontend.

    Parameters mirror the original script but the function returns a JSON serialisable
    structure instead of plotting or printing.
    """

    if etflist is None:
        etflist = [
            "BNDX",
            "SPSM",
            "SPMD",
            "SPLG",
            "VWO",
            "VEA",
            "MUB",
            "EMB",
        ]

    gridsize = 100

    # ------------------------------------------------------------------
    # 1) Data preparation
    # ------------------------------------------------------------------
    cdf = df[(df["ym"] >= startdate) & (df["ym"] <= enddate)].copy()
    useretfL = etflist + ["Mkt-RF", "RF2", "year", "month", "ym"]
    cdf = cdf[useretfL]

    if not maxuse:
        stacked = cdf[etflist].stack().reset_index()
        stacked.columns = ["row", "ticker", "value"]
        valid_obs = stacked.dropna()

        first_obs = valid_obs.iloc[0]
        first_valids = {
            etf: cdf[etf].first_valid_index()
            for etf in etflist
            if cdf[etf].first_valid_index() is not None
        }
        latest_start_row = first_valids[max(first_valids, key=first_valids.get)]
        startdate = int(cdf.loc[latest_start_row, "ym"])

        cdf = cdf.dropna()

    cdf.reset_index(drop=True, inplace=True)

    # Moments
    mu = cdf[etflist].mean()
    std = cdf[etflist].std()
    cov = cdf[etflist].cov()
    rf = float(cdf["RF2"].mean())

    descriptive_stats = [
        {"asset": a, "mean": float(mu[a]), "std": float(std[a]), "sr": float(mu[a] / std[a])}
        for a in etflist
    ]
    corr = cdf[etflist].corr().round(4)
    correlation_matrix = {
        "columns": list(corr.columns),
        "data": corr.to_dict(orient="records"),
    }

    # ------------------------------------------------------------------
    # 2) Optimisation helpers
    # ------------------------------------------------------------------
    def make_solvers(short_flag: int):
        if not short_flag:
            def solv_x(r, covdf, mu_vec):
                P = matrix(covdf.values)
                q = matrix(np.zeros(len(mu_vec)))
                G = -matrix(np.eye(len(mu_vec)))
                h = matrix(0.0, (len(mu_vec), 1))
                A = matrix(np.vstack((np.ones(len(mu_vec)), mu_vec.values)))
                b = matrix([1.0, r])
                solvers.options["show_progress"] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol["x"]).flatten()

            def solv_minvar(covdf, _):
                P = matrix(covdf.values)
                q = matrix(np.zeros(covdf.shape[0]))
                G = -matrix(np.eye(covdf.shape[0]))
                h = matrix(0.0, (covdf.shape[0], 1))
                A = matrix(1.0, (1, covdf.shape[0]))
                b = matrix(1.0)
                solvers.options["show_progress"] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol["x"]).flatten()

            def solv_maxret(mu_vec, _):
                c = -matrix(mu_vec.values)
                G = matrix(np.vstack((np.ones(len(mu_vec)), -np.eye(len(mu_vec)))))
                h = matrix(np.vstack((np.array([[1]]), np.zeros((len(mu_vec), 1)))))
                solvers.options["show_progress"] = False
                sol = solvers.lp(c, G, h)
                return np.array(sol["x"]).flatten()
        else:
            def solv_x(r, covdf, mu_vec):
                P = matrix(covdf.values)
                q = matrix(np.zeros(len(mu_vec)))
                G = -matrix(np.eye(len(mu_vec)))
                h = matrix(1.0, (len(mu_vec), 1))
                A = matrix(np.vstack((np.ones(len(mu_vec)), mu_vec.values)))
                b = matrix([1.0, r])
                solvers.options["show_progress"] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol["x"]).flatten()

            def solv_minvar(covdf, _):
                P = matrix(covdf.values)
                q = matrix(np.zeros(covdf.shape[0]))
                G = -matrix(np.eye(covdf.shape[0]))
                h = matrix(1.0, (covdf.shape[0], 1))
                A = matrix(1.0, (1, covdf.shape[0]))
                b = matrix(1.0)
                solvers.options["show_progress"] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol["x"]).flatten()

            def solv_maxret(mu_vec, _):
                c = -matrix(mu_vec.values)
                G = matrix(np.vstack((np.ones(len(mu_vec)), -np.eye(len(mu_vec)))))
                h = matrix(np.vstack((np.array([[1]]), np.zeros((len(mu_vec), 1)))))
                solvers.options["show_progress"] = False
                sol = solvers.lp(c, G, h)
                return np.array(sol["x"]).flatten()

        return solv_x, solv_minvar, solv_maxret

    solv_x, solv_minvar, solv_maxret = make_solvers(short)

    # ------------------------------------------------------------------
    # 3) Standard MV optimisation
    # ------------------------------------------------------------------
    minv = solv_minvar(cov, etflist)
    maxv = solv_maxret(mu, etflist)
    retspace = np.linspace(mu.dot(minv), mu.dot(maxv), gridsize)
    weightlist = [solv_x(r, cov, mu) for r in retspace]
    stdlist = [np.sqrt(w.dot(cov.values).dot(w)) for w in weightlist]
    srlist = [sharpe_ratio(w, mu, cov, rf) for w in weightlist]
    maxSRW = int(np.argmax(srlist))

    standard_efficient_frontier = [
        {"x": float(s * np.sqrt(12)), "y": float(r * 12)}
        for s, r in zip(stdlist, retspace)
    ]
    etf_points = [
        {
            "x": float(np.sqrt(cdf[t].var()) * np.sqrt(12)),
            "y": float(mu[t] * 12),
            "label": t,
        }
        for t in etflist
    ]
    standard_max_sr = {
        "x": float(stdlist[maxSRW] * np.sqrt(12)),
        "y": float(retspace[maxSRW] * 12),
    }
    standard_allocation_stack = [
        {
            "x": float(np.sqrt(w.dot(cov.values).dot(w)) * np.sqrt(12)),
            "allocations": {etflist[i]: float(w[i]) for i in range(len(etflist))},
        }
        for w in weightlist
    ]
    standard_weights = [
        {"asset": etflist[i], "weight": float(weightlist[maxSRW][i] * 100)}
        for i in range(len(etflist))
    ]
    standard_pie = {
        "labels": [w["asset"] for w in standard_weights],
        "values": [w["weight"] for w in standard_weights],
    }

    # ------------------------------------------------------------------
    # 4) Robust MV via Monte Carlo simulation
    # ------------------------------------------------------------------
    if not normal:
        simw = np.zeros((gridsize, len(etflist)))
        Nsim = 200
        random.seed(12345)
        for _ in range(Nsim):
            simdata = np.random.multivariate_normal(mu.values, cov.values, len(cdf))
            simdf = pd.DataFrame(simdata, columns=etflist)
            mu_s = simdf.mean()
            cov_s = simdf.cov()

            min_w = solv_minvar(cov_s, etflist)
            max_w = solv_maxret(mu_s, etflist)
            retspace_s = np.linspace(mu_s.dot(min_w), mu_s.dot(max_w), gridsize)
            simw += np.array([solv_x(r, cov_s, mu_s) for r in retspace_s])

        simw /= Nsim
        efstd = [np.sqrt(w.dot(cov.values).dot(w)) * np.sqrt(12) for w in simw]
        efret = [mu.dot(w) * 12 for w in simw]
        SRlist = [sharpe_ratio(w, mu, cov, rf) for w in simw]
        idx_rob = int(np.argmax(SRlist))
        robw = simw[idx_rob]

        robust_efficient_frontier = [
            {"x": float(efstd[i]), "y": float(efret[i])} for i in range(gridsize)
        ]
        robust_allocation_stack = [
            {
                "x": float(efstd[i]),
                "allocations": {etflist[j]: float(simw[i][j]) for j in range(len(etflist))},
            }
            for i in range(gridsize)
        ]
        robust_max_sr = {"x": float(efstd[idx_rob]), "y": float(efret[idx_rob])}
        robust_weights = [
            {"asset": etflist[i], "weight": float(robw[i] * 100)}
            for i in range(len(etflist))
        ]
        robust_pie = {
            "labels": [w["asset"] for w in robust_weights],
            "values": [w["weight"] for w in robust_weights],
        }
    else:
        # Robust set equals standard when normal == 1
        robust_efficient_frontier = standard_efficient_frontier
        robust_allocation_stack = standard_allocation_stack
        robust_max_sr = standard_max_sr
        robust_weights = standard_weights
        robust_pie = standard_pie

    return {
        "descriptive_stats": descriptive_stats,
        "correlation_matrix": correlation_matrix,
        "standard_mv": {
            "efficient_frontier": standard_efficient_frontier,
            "etf_points": etf_points,
            "max_sr_point": standard_max_sr,
            "allocation_stack": standard_allocation_stack,
            "weights": standard_weights,
            "pie_chart": standard_pie,
        },
        "robust_mv": {
            "efficient_frontier": robust_efficient_frontier,
            "max_sr_point": robust_max_sr,
            "allocation_stack": robust_allocation_stack,
            "weights": robust_weights,
            "pie_chart": robust_pie,
        },
        "short": int(short),
    }

