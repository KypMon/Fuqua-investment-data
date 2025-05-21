import numpy as np
import pandas as pd


from cvxopt import matrix, solvers

def sharpe_ratio(w, mean_ret, cov, rf):
    return (w @ mean_ret - rf) / np.sqrt(w.T @ cov @ w)

def mv(df,
       etflist=['BNDX','SPSM','SPMD','SPLG','VWO','VEA','MUB','EMB'],
       short=0,
       maxuse=1,
       normal=1,
       startdate=199302,
       enddate=202312):
    """
    输入：
      df: 包含列 ['year','month','ym','ticker_new',...] 的 DataFrame
      etflist: 要做 MV 的 ETF 列名列表
      short:   0=禁止空头，1=允许空头
      maxuse:  0=dropna，1=keep all
      normal:  1=标准MV，0=Robust MV
      startdate,enddate: YYYYMM 整型

    返回：一个 dict，包含
      descriptive_stats, correlation_matrix,
      efficient_frontier, etf_points, max_sr_point,
      allocation_stack, robust_weights, pie_chart, short
    """

    gridsize = 100


    # 1) 过滤并选列
    cdf = df[(df['ym'] >= startdate) & (df['ym'] <= enddate)].copy()
    cols = etflist + ['Mkt-RF','RF','year','month','ym']
    cdf = cdf[cols]
    if not maxuse:
        cdf.dropna(inplace=True)
    cdf.reset_index(drop=True, inplace=True)

    # Calculate the original moments
    meandf = cdf[etflist].mean()
    covdf = cdf[etflist].cov()
    stddf = np.sqrt(cdf[etflist].var())
    assetsrdf = meandf/stddf

    # 2) 描述性统计
    meandf = cdf[etflist].mean()
    stddf = cdf[etflist].std()
    srdf = meandf / stddf
    descriptive_stats = [
        {
            "asset": a,
            "mean": float(meandf[a]),
            "std":  float(stddf[a]),
            "sr":   float(srdf[a])
        }
        for a in etflist
    ]

    # 3) 相关矩阵
    corr = cdf[etflist].corr().round(4)
    correlation_matrix = {
        "columns": list(corr.columns),
        "data":    corr.to_dict(orient="records")
    }

    # 4) 无风险利率
    rf = float(cdf['RF'].mean())

    # 5) 生成优化器
    def make_solvers(short_flag):
        if short_flag == 0:
            # 不允许空头
            def solv_x(r, covm, mu):
                P = matrix(covm.values)
                q = matrix(np.zeros(len(mu)))
                G = -matrix(np.eye(len(mu)))
                h = matrix(0.0, (len(mu),1))
                A = matrix(np.vstack((np.ones(len(mu)), mu.values)))
                b = matrix([1.0, r])
                solvers.options['show_progress'] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol['x']).flatten()
            def solv_minvar(covm):
                P = matrix(covm.values)
                q = matrix(np.zeros(covm.shape[0]))
                G = -matrix(np.eye(covm.shape[0]))
                h = matrix(0.0, (covm.shape[0],1))
                A = matrix(1.0, (1, covm.shape[0]))
                b = matrix(1.0)
                solvers.options['show_progress'] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol['x']).flatten()
            def solv_maxret(mu):
                c = -matrix(mu.values)
                G = matrix(np.vstack((np.ones(len(mu)), -np.eye(len(mu)))))
                h = matrix(np.vstack((np.array([[1]]), np.zeros((len(mu),1)))))
                solvers.options['show_progress'] = False
                sol = solvers.lp(c, G, h)
                return np.array(sol['x']).flatten()
        else:
            # 允许空头
            def solv_x(r, covm, mu):
                P = matrix(covm.values)
                q = matrix(np.zeros(len(mu)))
                G = -matrix(np.eye(len(mu)))
                h = matrix(1.0, (len(mu),1))
                A = matrix(np.vstack((np.ones(len(mu)), mu.values)))
                b = matrix([1.0, r])
                solvers.options['show_progress'] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol['x']).flatten()
            def solv_minvar(covm):
                P = matrix(covm.values)
                q = matrix(np.zeros(covm.shape[0]))
                G = -matrix(np.eye(covm.shape[0]))
                h = matrix(1.0, (covm.shape[0],1))
                A = matrix(1.0, (1, covm.shape[0]))
                b = matrix(1.0)
                solvers.options['show_progress'] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol['x']).flatten()
            def solv_maxret(mu):
                c = -matrix(mu.values)
                G = matrix(np.vstack((np.ones(len(mu)), -np.eye(len(mu)))))
                h = matrix(np.vstack((np.array([[1]]), np.zeros((len(mu),1)))))
                solvers.options['show_progress'] = False
                sol = solvers.lp(c, G, h)
                return np.array(sol['x']).flatten()
        return solv_x, solv_minvar, solv_maxret

    solv_x, solv_minvar, solv_maxret = make_solvers(int(short))

    # 6) 计算标准 MV / Robust MV
    # 6.1 标准 MV
    minvar_w = solv_minvar(covdf)
    maxret_w = solv_maxret(meandf)

    minret = float(meandf.dot(minvar_w))
    maxret = float(meandf.dot(maxret_w))
    retspace = np.linspace(minret, maxret, gridsize)

    weightlist = [solv_x(r, covdf, meandf) for r in retspace]
    stdlist    = [np.sqrt(w @ covdf.values @ w) for w in weightlist]
    srlist     = [sharpe_ratio(w, meandf, covdf, rf) for w in weightlist]

    # Maximum Sharpe Ratio Portfolio
    maxSRW  = np.argmax(srlist)
    maxSR_ret = weightlist[maxSRW]@meandf
    maxSR_std = np.sqrt(weightlist[maxSRW]@covdf@weightlist[maxSRW])

    perctw = weightlist[maxSRW]

    print(perctw)

    max_idx = int(np.argmax(srlist))
    max_sr_point = {
        "x": float(stdlist[max_idx]),
        "y": float(retspace[max_idx] * 12)
    }

    efficient_frontier = [
        {"x": float(stdlist[i]), "y": float(retspace[i] * 12)}
        for i in range(len(stdlist))
    ]

    etf_points = []
    for t in etflist:
        sigma = float(np.sqrt(cdf[t].var()) * np.sqrt(12))
        mu_ann = float(meandf[t] * 12)
        etf_points.append({"x": sigma, "y": mu_ann, "label": t})

    allocation_stack = []
    for w in weightlist:
        allocation_stack.append({
            "x": float(np.sqrt(w @ covdf.values @ w) * np.sqrt(12)),
            "allocations": {etflist[i]: float(w[i]) for i in range(len(etflist))}
        })

    # 7) Robust weights
    if normal:
        robw = weightlist[max_idx]
    else:
        # 简单重采样
        simw_sum = np.zeros((gridsize, len(etflist)))
        for i in range(gridsize):
            simret = np.random.multivariate_normal(meandf.values,
                                                   covdf.values,
                                                   len(cdf))
            simdf  = pd.DataFrame(simret, columns=etflist)
            mu_s   = simdf.mean()
            cov_s  = simdf.cov()
            wlist  = [solv_x(r, cov_s, mu_s) for r in retspace]
            simw_sum[i,:] = np.vstack(wlist).sum(axis=0)
        simw_avg = simw_sum / gridsize
        srsim = [sharpe_ratio(simw_avg[i,:], meandf, covdf, rf)
                 for i in range(gridsize)]
        best_i = int(np.argmax(srsim))
        robw = simw_avg[best_i,:]

    robust_weights = [
        {"asset": etflist[i], "weight": weightlist[maxSRW][i] * 100 }
        for i in range(len(etflist))
    ]

    pie_chart = {
        "labels": [r["asset"] for r in robust_weights],
        "values": [r["weight"] for r in robust_weights]
    }

    return {
        "descriptive_stats":  descriptive_stats,
        "correlation_matrix": correlation_matrix,
        "efficient_frontier": efficient_frontier,
        "etf_points":         etf_points,
        "max_sr_point":       max_sr_point,
        "allocation_stack":   allocation_stack,
        "robust_weights":     robust_weights,
        "pie_chart":          pie_chart,
        "short":              int(short)
    }