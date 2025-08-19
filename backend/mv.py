import numpy as np
import pandas as pd
import random 

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
    gridsize = 100

    # 1) 筛选时间 & 列
    cdf = df[(df['ym'] >= startdate) & (df['ym'] <= enddate)].copy()
    cols = etflist + ['Mkt-RF','RF','year','month','ym']
    cdf = cdf[cols]
    if not maxuse:
        cdf.dropna(inplace=True)
    cdf.reset_index(drop=True, inplace=True)

    # 2) 描述性统计
    meandf = cdf[etflist].mean()
    stddf = cdf[etflist].std()
    srdf   = meandf / stddf
    descriptive_stats = [
        {'asset': a, 'mean': float(meandf[a]), 'std': float(stddf[a]), 'sr': float(srdf[a])}
        for a in etflist
    ]

    # 3) 相关矩阵
    corr = cdf[etflist].corr().round(4)
    correlation_matrix = {
        'columns': list(corr.columns),
        'data': corr.to_dict(orient='records')
    }

    # 4) 无风险利率
    rf = float(cdf['RF'].mean())

    # 5) 构造 CVXOPT 求解器
    def make_solvers(short_flag):
        if not short_flag:
            def solv_x(r, covdf, mu):
                P = matrix(covdf.values)
                q = matrix(np.zeros(len(mu)))
                G = -matrix(np.eye(len(mu)))
                h = matrix(0.0, (len(mu),1))
                A = matrix(np.vstack((np.ones(len(mu)), mu.values)))
                b = matrix([1.0, r])
                solvers.options['show_progress'] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol['x']).flatten()

            def solv_minvar(covdf, _):
                P = matrix(covdf.values)
                q = matrix(np.zeros(covdf.shape[0]))
                G = -matrix(np.eye(covdf.shape[0]))
                h = matrix(0.0, (covdf.shape[0],1))
                A = matrix(1.0, (1, covdf.shape[0]))
                b = matrix(1.0)
                solvers.options['show_progress'] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol['x']).flatten()

            def solv_maxret(mu, _):
                c = -matrix(mu.values)
                G = matrix(np.vstack((np.ones(len(mu)), -np.eye(len(mu)))))
                h = matrix(np.vstack((np.array([[1]]), np.zeros((len(mu),1)))))
                solvers.options['show_progress'] = False
                sol = solvers.lp(c, G, h)
                return np.array(sol['x']).flatten()
        else:
            def solv_x(r, covdf, mu):
                P = matrix(covdf.values)
                q = matrix(np.zeros(len(mu)))
                G = -matrix(np.eye(len(mu)))
                h = matrix(1.0, (len(mu),1))
                A = matrix(np.vstack((np.ones(len(mu)), mu.values)))
                b = matrix([1.0, r])
                solvers.options['show_progress'] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol['x']).flatten()

            def solv_minvar(covdf, _):
                P = matrix(covdf.values)
                q = matrix(np.zeros(covdf.shape[0]))
                G = -matrix(np.eye(covdf.shape[0]))
                h = matrix(1.0, (covdf.shape[0],1))
                A = matrix(1.0, (1, covdf.shape[0]))
                b = matrix(1.0)
                solvers.options['show_progress'] = False
                sol = solvers.qp(P, q, G, h, A, b)
                return np.array(sol['x']).flatten()

            def solv_maxret(mu, _):
                c = -matrix(mu.values)
                G = matrix(np.vstack((np.ones(len(mu)), -np.eye(len(mu)))))
                h = matrix(np.vstack((np.array([[1]]), np.zeros((len(mu),1)))))
                solvers.options['show_progress'] = False
                sol = solvers.lp(c, G, h)
                return np.array(sol['x']).flatten()

        return solv_x, solv_minvar, solv_maxret

    solv_x, solv_minvar, solv_maxret = make_solvers(short)

    # 6) Standard MV Portfolio
    covdf = cdf[etflist].cov()
    minv = solv_minvar(covdf, etflist)
    maxv = solv_maxret(meandf, etflist)
    retspace_m = np.linspace(meandf.dot(minv), meandf.dot(maxv), gridsize)
    weightlist = [solv_x(r, covdf, meandf) for r in retspace_m]
    stdlist_m   = [np.sqrt(w.dot(covdf.values).dot(w)) for w in weightlist]
    srlist      = [sharpe_ratio(w, meandf, covdf, rf) for w in weightlist]
    maxSRW      = int(np.argmax(srlist))

    standard_efficient_frontier = [
        {'x': float(s * np.sqrt(12)), 'y': float(r * 12)}
        for s,r in zip(stdlist_m, retspace_m)
    ]
    etf_points = [
        {'x': float(np.sqrt(cdf[t].var())*np.sqrt(12)),
         'y': float(meandf[t]*12),
         'label': t}
        for t in etflist
    ]
    standard_max_sr = {
        'x': float(stdlist_m[maxSRW]*np.sqrt(12)),
        'y': float(retspace_m[maxSRW]*12)
    }
    standard_allocation_stack = [
        {'x': float(np.sqrt(w.dot(covdf.values).dot(w))*np.sqrt(12)),
         'allocations': {etflist[i]: float(w[i]) for i in range(len(etflist))}}
        for w in weightlist
    ]
    standard_weights = [
        {'asset': etflist[i], 'weight': float(weightlist[maxSRW][i]*100)}
        for i in range(len(etflist))
    ]
    standard_pie = {
        'labels': [w['asset'] for w in standard_weights],
        'values': [w['weight'] for w in standard_weights]
    }

    # 7) Robust MV Portfolio (if normal==0 做 Monte‐Carlo，否则直接复用 standard)
    if not normal:
        simw = np.zeros((gridsize, len(etflist)))
        Nsim = 100
        random.seed(123)
        for _ in range(Nsim):
            sample = np.random.multivariate_normal(meandf.values, covdf.values, size=len(cdf))
            simdf  = pd.DataFrame(sample, columns=etflist)
            mu_s   = simdf.mean()
            cov_s  = simdf.cov()
            for j, r in enumerate(retspace_m):
                simw[j] += solv_x(r, cov_s, mu_s)
        simw /= Nsim
        sr_sim = [sharpe_ratio(simw[j], meandf, covdf, rf) for j in range(gridsize)]
        idx_rob = int(np.argmax(sr_sim))
        robw = simw[idx_rob]

        # Robust frontier for consistency（可选渲染）
        robust_efficient_frontier = [
            {'x': float(np.sqrt(simw[j].dot(covdf.values).dot(simw[j]))*np.sqrt(12)),
             'y': float(retspace_m[j]*12)}
            for j in range(gridsize)
        ]
        robust_allocation_stack = [
            {'x': float(np.sqrt(simw[j].dot(covdf.values).dot(simw[j]))*np.sqrt(12)),
             'allocations': {etflist[i]: float(simw[j][i]) for i in range(len(etflist))}}
            for j in range(gridsize)
        ]
    else:
        # 如果 normal==1，就把 robust 直接设为 standard
        idx_rob = maxSRW
        robw = weightlist[maxSRW]
        robust_efficient_frontier = standard_efficient_frontier
        robust_allocation_stack    = standard_allocation_stack

    robust_max_sr = {
        'x': float(np.sqrt(robw.dot(covdf.values).dot(robw))*np.sqrt(12)),
        'y': float(retspace_m[idx_rob]*12)
    }
    robust_weights = [
        {'asset': etflist[i], 'weight': float(robw[i]*100)}
        for i in range(len(etflist))
    ]
    robust_pie = {
        'labels': [w['asset'] for w in robust_weights],
        'values': [w['weight'] for w in robust_weights]
    }

    return {
        "descriptive_stats":     descriptive_stats,
        "correlation_matrix":    correlation_matrix,
        "standard_mv": {
            "efficient_frontier": standard_efficient_frontier,
            "etf_points":         etf_points,
            "max_sr_point":       standard_max_sr,
            "allocation_stack":   standard_allocation_stack,
            "weights":            standard_weights,
            "pie_chart":          standard_pie
        },
        "robust_mv": {
            "efficient_frontier": robust_efficient_frontier,
            "max_sr_point":       robust_max_sr,
            "allocation_stack":   robust_allocation_stack,
            "weights":            robust_weights,
            "pie_chart":          robust_pie
        },
        "short": int(short)
    }