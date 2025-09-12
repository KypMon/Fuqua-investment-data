import numpy as np
import pandas as pd
import random

from cvxopt import matrix, solvers


def sharpe_ratio(x: np.ndarray, meandf: pd.Series, covdf: pd.DataFrame, rf: float) -> float:
    """Sharpe ratio for portfolio weights ``x``."""
    return (x @ meandf - rf) / np.sqrt(x.T @ covdf @ x)


def mv(
    df: pd.DataFrame,
    etflist=None,
    short: int = 0,
    maxuse: int = 1,
    normal: int = 1,
    startdate: int = 199302,
    enddate: int = 202312,
):
    """Mean-variance routine matching the original backend logic.

    All calculations follow the script-based implementation but plotting and
    printing are removed. Instead the function returns a dictionary ready for
    JSON serialisation so the frontend can render the results.
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

    print("123")

    try:
        cdf = df[(df["ym"] >= startdate) & (df["ym"] <= enddate)]
        useretfL = etflist + ["Mkt-RF", "RF", "year", "month", "ym"]
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

            latest_start_ticker = max(first_valids, key=first_valids.get)
            latest_start_row = first_valids[latest_start_ticker]
            startdate = int(cdf.loc[latest_start_row, "ym"])

            cdf = cdf.dropna()

        cdf.reset_index(inplace=True)

        meandf = cdf[etflist].mean()
        covdf = cdf[etflist].cov()
        stddf = np.sqrt(cdf[etflist].var())
        assetsrdf = meandf / stddf
        descriptive_stats = [
            {
                "asset": etflist[i],
                "mean": float(meandf[i]),
                "std": float(stddf[i]),
                "sr": float(assetsrdf[i]),
            }
            for i in range(len(etflist))
        ]
        corr = cdf[etflist].corr().round(4)
        correlation_matrix = {
            "columns": list(corr.columns),
            "data": corr.to_dict(orient="records"),
        }

        rf = cdf["RF"].mean()

        print("123")

        if not short:
            def solv_x(r, covdf, meandf, etflist):
                covmat = matrix(covdf.values)
                P = matrix(np.zeros(len(etflist)))
                G = -matrix(np.eye(len(etflist)))
                h = matrix(0.0, (len(etflist), 1))
                A = matrix(np.vstack((np.ones(len(etflist)), meandf)))
                b = matrix([1.0, r])
                solvers.options['show_progress'] = False
                solv = solvers.qp(covmat, P, G, h, A, b)
                x = np.array(solv['x']).flatten()
                return x

            def solv_minvar(simcovdf, etflist):
                covmat = matrix(simcovdf.values)
                P = matrix(np.zeros(len(etflist)))
                G = -matrix(np.eye(len(etflist)))
                h = matrix(0.0, (len(etflist), 1))
                A = matrix(1.0, (1, len(etflist)))
                b = matrix(1.0)
                solvers.options['show_progress'] = False
                solv = solvers.qp(covmat, P, G, h, A, b)
                x = np.array(solv['x']).flatten()
                return x

            def solv_maxret(simmeandf, etflist):
                c = -matrix(simmeandf.values)
                G = matrix(np.vstack((np.ones(len(etflist)), -np.eye(len(etflist)))))
                h = matrix(np.vstack((np.array([[1]]), np.zeros((len(etflist), 1)))))
                solvers.options['show_progress'] = False
                solv = solvers.lp(c, G, h)
                x = np.array(solv['x']).flatten()
                return x
        else:
            def solv_x(r, covdf, meandf, etflist):
                covmat = matrix(covdf.values)
                P = matrix(np.zeros(len(etflist)))
                G = -matrix(np.eye(len(etflist)))
                h = matrix(1.0, (len(etflist), 1))
                A = matrix(np.vstack((np.ones(len(etflist)), meandf)))
                b = matrix([1.0, r])
                solvers.options['show_progress'] = False
                solv = solvers.qp(covmat, P, G, h, A, b)
                x = np.array(solv['x']).flatten()
                return x

            def solv_minvar(simcovdf, etflist):
                covmat = matrix(simcovdf.values)
                P = matrix(np.zeros(len(etflist)))
                G = -matrix(np.eye(len(etflist)))
                h = matrix(1.0, (len(etflist), 1))
                A = matrix(1.0, (1, len(etflist)))
                b = matrix(1.0)
                solvers.options['show_progress'] = False
                solv = solvers.qp(covmat, P, G, h, A, b)
                x = np.array(solv['x']).flatten()
                return x

            def solv_maxret(simmeandf, etflist):
                c = -matrix(simmeandf.values)
                G = matrix(np.vstack((np.ones(len(etflist)), -np.eye(len(etflist)))))
                h = matrix(np.vstack((np.array([[1]]), np.zeros((len(etflist), 1)))))
                solvers.options['show_progress'] = False
                solv = solvers.lp(c, G, h)
                x = np.array(solv['x']).flatten()
                return x

        minvar_w = solv_minvar(covdf, etflist)
        maxret_w = solv_maxret(meandf, etflist)
        minret = meandf @ minvar_w
        maxret = meandf @ maxret_w
        retspace = np.linspace(minret, maxret, gridsize)

        weightlist = [solv_x(i, covdf, meandf, etflist) for i in retspace]
        stdlist = [np.sqrt(i @ covdf @ i) for i in weightlist]
        SRlist = [sharpe_ratio(i, meandf, covdf, rf) for i in weightlist]

        maxSRW = np.argmax(SRlist)
        maxSR_ret = weightlist[maxSRW] @ meandf
        maxSR_std = np.sqrt(weightlist[maxSRW] @ covdf @ weightlist[maxSRW])

        standard_efficient_frontier = [
            {"x": float(std * np.sqrt(12)), "y": float(ret * 12)}
            for std, ret in zip(stdlist, retspace)
        ]
        etf_points = [
            {
                "x": float(stddf[i] * np.sqrt(12)),
                "y": float(meandf[i] * 12),
                "label": etflist[i],
            }
            for i in range(len(etflist))
        ]
        standard_max_sr = {
            "x": float(maxSR_std * np.sqrt(12)),
            "y": float(maxSR_ret * 12),
        }
        standard_allocation_stack = [
            {
                "x": float(np.sqrt(w @ covdf @ w) * np.sqrt(12)),
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

        if not normal:
            robw = np.zeros(len(etflist))
            simwdf = np.zeros(gridsize)

            Nsim = 200
            iter = 0
            random.seed(12345)
            while iter < Nsim:
                simdata = np.random.multivariate_normal(
                    meandf.values, covdf.values, len(cdf)
                )
                simdf = pd.DataFrame(simdata, columns=etflist)
                simmeandf = simdf.mean()
                simcovdf = simdf.cov()

                def solv_x(r, simcovdf, simmeandf, etflist):
                    covmat = matrix(simcovdf.values)
                    P = matrix(np.zeros(len(etflist)))
                    G = -matrix(np.eye(len(etflist)))
                    h = matrix(0.0, (len(etflist), 1))
                    A = matrix(np.vstack((np.ones(len(etflist)), simmeandf)))
                    b = matrix([1.0, r])
                    solvers.options["show_progress"] = False
                    solv = solvers.qp(covmat, P, G, h, A, b)
                    x = np.array(solv["x"]).flatten()
                    return x

                def solv_minvar(simcovdf, etflist):
                    covmat = matrix(simcovdf.values)
                    P = matrix(np.zeros(len(etflist)))
                    G = -matrix(np.eye(len(etflist)))
                    h = matrix(0.0, (len(etflist), 1))
                    A = matrix(1.0, (1, len(etflist)))
                    b = matrix(1.0)
                    solvers.options["show_progress"] = False
                    solv = solvers.qp(covmat, P, G, h, A, b)
                    x = np.array(solv["x"]).flatten()
                    return x

                def solv_maxret(simmeandf, etflist):
                    c = -matrix(simmeandf.values)
                    G = matrix(
                        np.vstack((np.ones(len(etflist)), -np.eye(len(etflist))))
                    )
                    h = matrix(
                        np.vstack((np.array([[1]]), np.zeros((len(etflist), 1))))
                    )
                    solvers.options["show_progress"] = False
                    solv = solvers.lp(c, G, h)
                    x = np.array(solv["x"]).flatten()
                    return x

                minvar_w = solv_minvar(simcovdf, etflist)
                maxret_w = solv_maxret(simmeandf, etflist)

                minret = simmeandf @ minvar_w
                maxret = simmeandf @ maxret_w
                retspace = np.linspace(minret, maxret, gridsize)

                weightlist = [solv_x(i, simcovdf, simmeandf, etflist) for i in retspace]
                simwdf = [a + b for a, b in zip(simwdf, weightlist)]

                iter = iter + 1

            simwdf = [w / Nsim for w in simwdf]

            efstd = [np.sqrt(12 * w @ covdf @ w) for w in simwdf]
            efret = [12 * (w @ meandf) for w in simwdf]
            SRlist = [sharpe_ratio(w, meandf, covdf, rf) for w in simwdf]
            maxSR = np.argmax(SRlist)
            maxSR_ret = efret[maxSR]
            maxSR_std = efstd[maxSR]
            robw = simwdf[maxSR]

            cml_std = np.linspace(0, efstd[-1], gridsize)
            cml_ret = [std * (maxSR_ret - rf * 12) / maxSR_std + rf * 12 for std in cml_std]

            robust_efficient_frontier = [
                {"x": float(efstd[i]), "y": float(efret[i])} for i in range(gridsize)
            ]
            robust_allocation_stack = [
                {
                    "x": float(efstd[i]),
                    "allocations": {etflist[j]: float(simwdf[i][j]) for j in range(len(etflist))},
                }
                for i in range(gridsize)
            ]
            robust_max_sr = {"x": float(maxSR_std), "y": float(maxSR_ret)}
            robust_weights = [
                {"asset": etflist[i], "weight": float(robw[i] * 100)}
                for i in range(len(etflist))
            ]
            robust_pie = {
                "labels": [w["asset"] for w in robust_weights],
                "values": [w["weight"] for w in robust_weights],
            }
        else:
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
    except Exception as e:
        print(e)
        return {"error": "error"}

