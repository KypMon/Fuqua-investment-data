
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from contextlib import redirect_stdout
from flask_cors import CORS
import yfinance as yf
import calendar
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import random 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


app = Flask(
    __name__,
    static_url_path="/static",
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)
CORS(app)
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# -------- Portfolio Logic (from robust_mv10.py) --------
## This code is for the MV playground 3 ## 

# %%
plt.rcParams['figure.figsize'] = [15, 5]
from cvxopt import matrix, solvers
import cvxpy as cp
from tabulate import tabulate

# Global Setup
plt.rcParams['figure.figsize'] = [15, 5]
ff_file = 'F-F_Research_Data_Factors.csv'
etf_file = 'stocks_mf_ETF_data_final.csv'


# %% 
def get_data(file_name):
    path = os.path.join(os.getcwd(), file_name)
    # ETF
    try:
        df = pd.read_csv(path)
        df = df.pivot_table(index=['year', 'month'], columns = 'ticker_new', values='ret')
        df.reset_index(inplace=True)

        df.drop(columns={'RF'}, inplace = True)
        return df
    # FF
    except pd.errors.ParserError as e:
        df = pd.read_csv(path, skiprows=3)
        first_non_numeric_index = None
        for index, value in df['Unnamed: 0'].items():
            if not is_numeric(value):
                first_non_numeric_index = index
                break
        
        df = df[:first_non_numeric_index]
        df['year'] = df['Unnamed: 0'].astype(str).str[:4]
        df['month'] = df['Unnamed: 0'].astype(str).str[4:6]
        df.drop(columns=['Unnamed: 0'], inplace=True)

        for column in df.columns:
            if column != 'year' and column != 'month':
                df[column] = df[column].astype(float)
            else: 
                
                df[column] = df[column].astype(int)
        df['RF'] = df['RF'] / 100
        df.drop(columns={'SMB', 'HML'}, inplace = True)
        return df

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
def get_and_merge(ff_file, etf_file):
    ffdf = get_data(ff_file)


    etfdf = get_data(etf_file)

    df = pd.merge(etfdf, ffdf, on=['year', 'month'], how='inner')
    df['ym'] = df['year']*100 + df['month']
    df['ym'] = df['ym'].astype(int)

    return df

# Global Data
df = get_and_merge(ff_file, etf_file)

# Calculate sharpe ratio
def sharpe_ratio(x, meandf, covdf, rf): 
    sp = (x@meandf-rf)/np.sqrt(x.T@covdf@x)
    return sp

def mv(df, etflist = ['BNDX', 'SPSM', 'SPMD', 'SPLG', 'VWO', 'VEA', 'MUB', 'EMB'], short = 0, maxuse = 1, normal = 1, startdate = 199302, enddate = 202312):

    gridsize = 100

    try: 
        cdf = df[(df['ym'] >= startdate) & (df['ym'] <= enddate)]

        useretfL = etflist + ['Mkt-RF', 'RF', 'year', 'month', 'ym']
        cdf = cdf[useretfL]
        
        # Indicating whether to use the maximum available data
        if not maxuse: 
            cdf = cdf.dropna()
        cdf.reset_index(inplace = True)
        
        # Calculate the original moments
        meandf = cdf[etflist].mean()
        covdf = cdf[etflist].cov()
        stddf = np.sqrt(cdf[etflist].var())
        assetsrdf = meandf/stddf
        print("Asset Descriptive Statistics: ")
        for i in range(len(etflist)): 
            print(f"Asset {i+1} - {etflist[i]}: Mean - {meandf[i].round(4)}, Std - {stddf[i].round(4)}, SR - {assetsrdf[i].round(4)}")
        print("Asset Correlation Matrix: ")
        print(cdf[etflist].corr())

        # Risk Free Rate
        rf = cdf['RF'].mean()
        
        # Short Selling option
        if not short: 
            shortchoice = 'w/o.'
        else: 
            shortchoice = 'w/.'
        
        # Standard MV Portfolio 
        if normal: 
            if not short: 
                # solve for optimal weight that minimize STD given return
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
                # Minimum Variance Portfolio 
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
                
                
                # Maximum Return Portfolio
                def solv_maxret(simmeandf, etflist): 
                    c = -matrix(simmeandf.values)
                    G = matrix(np.vstack((np.ones(len(etflist)), -np.eye(len(etflist)))))
                    h = matrix(np.vstack((np.array([[1]]), np.zeros((len(etflist), 1)))))
                    solvers.options['show_progress'] = False
                    solv = solvers.lp(c, G, h)
                    x = np.array(solv['x']).flatten()
                    return x
            else: 
                # solve for optimal weight that minimize STD given return, with short selling
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
                
            # Initiate the linspace of return
            minret = meandf@minvar_w
            maxret = meandf@maxret_w
            retspace = np.linspace(minret, maxret, gridsize)
            
            # Weight, Std, and SR calculation
            weightlist = [solv_x(i, covdf, meandf, etflist) for i in retspace]
            stdlist = [np.sqrt(i@covdf@i) for i in weightlist]
            SRlist = [sharpe_ratio(i, meandf, covdf, rf) for i in weightlist]
            
            # Maximum Sharpe Ratio Portfolio
            maxSRW  = np.argmax(SRlist)
            maxSR_ret = weightlist[maxSRW]@meandf
            maxSR_std = np.sqrt(weightlist[maxSRW]@covdf@weightlist[maxSRW])
            
            # Report the MV Portfolio Weight
            print("Max Sharpe Ratio Portfolio Weights: ")
            for i in range(len(etflist)): 
                perctw = weightlist[maxSRW][i] * 100
                print(f"Asset {i+1} - {etflist[i]}: {perctw.round(2)}%")
            if not short: 
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')

                # Create the pie chart
                wedges, texts, autotexts = ax.pie(weightlist[maxSRW], autopct='%1.1f%%',
                    shadow=False, startangle=140)
                ax.legend(wedges, etflist, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=len(etflist))
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')
                plt.title(f'Max Sharpe Ratio Portfolio Weights, {shortchoice} Short Selling, Date Range: {startdate}-{enddate}')
                plt.show()

            # Plot
            gl = min(min(stdlist), min(stddf)) * 0.7 * np.sqrt(12)
            gr = max(max(stdlist), max(stddf)) * 1.1 * np.sqrt(12)
            gu = max(max(retspace), max(meandf)) * 1.15 * 12
            gb = min(min(retspace), min(meandf)) * 0.7 * 12
            
            stdlist = [std * np.sqrt(12) for std in stdlist]
            retspace = retspace * 12
            maxSR_ret = maxSR_ret * 12
            maxSR_std = maxSR_std * np.sqrt(12)
            stddf = stddf * np.sqrt(12)
            meandf = meandf * 12
            
            plt.plot(figsize=(15,5))
            plt.plot(stdlist, retspace, linewidth = 1)
            plt.scatter(stddf, meandf, color='purple', marker='o', s=40)
            for i in range(len(etflist)): 
                plt.annotate(etflist[i], (stddf[i], meandf[i]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.scatter(maxSR_std, maxSR_ret, color='red', marker='*', s=110)
            plt.text(maxSR_std, maxSR_ret, s="MVP", horizontalalignment='right', verticalalignment='top', fontsize=10)
            plt.gca().set_xlim(left=0)
            plt.gca().set_ylim(bottom=0)
            plt.xlim(gl, gr)
            plt.ylim(gb, gu)
            plt.title(f'Standard MV Portfolio, {shortchoice} Short Selling, Date Range: {startdate}-{enddate}')
            print(shortchoice)
            plt.show()

            if not short: 
                colors = ['orange', 'blue', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow']
                colorlist = colors[:len(etflist)]
                fig, ax = plt.subplots(figsize=(12, 6))
                bottom = np.zeros_like(stdlist) 
                allocations = pd.DataFrame(weightlist, columns = etflist)
                for i, e in enumerate(allocations.columns):
                    ax.fill_between(stdlist, bottom, bottom + allocations[e], label = e, color=colorlist[i], alpha=0.5)
                    bottom += allocations[e]  
                plt.title(f'Efficient Frontier Transition Map, Date Range: {startdate}-{enddate}')
                plt.xlabel('Standard Deviation')
                plt.ylabel('Allocation')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(etflist))
                plt.show()
            
            print("Efficient Frontier Portfolios:")
            
            efpdf1 = pd.DataFrame(weightlist, columns = etflist)
            efpdf2 = pd.DataFrame({'Return': retspace, 'Std': stdlist, 'SR': SRlist})
            efpdf = pd.concat([efpdf1, efpdf2], axis=1)
            efpdf = efpdf.round(4)
            efpdf.index = efpdf.index + 1
            efpdf.index.name = '#'
            print(tabulate(efpdf, headers='keys', tablefmt='github'))

            
            
        # Robust MV Portfolio
        else: 
            robw = np.zeros(len(etflist))
            simwdf = np.zeros(gridsize)
            
            # Simulation Parameters Set Up
            Nsim = 5000
            iter = 0
            random.seed(123)
            while iter < Nsim: 
                if iter % 10 == 0 and iter > 1: 
                    print(f"Completed {round(iter*100/Nsim)}%")
                simdata = np.random.multivariate_normal(meandf.values, covdf.values, len(cdf))
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
                    solvers.options['show_progress'] = False
                    solv = solvers.qp(covmat, P, G, h, A, b)
                    x = np.array(solv['x']).flatten()
                    return x
                
                # Minimum Variance Portfolio 
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
                minvar_w = solv_minvar(simcovdf, etflist)
                
                # Maximum Return Portfolio
                def solv_maxret(simmeandf, etflist): 
                    c = -matrix(simmeandf.values)
                    G = matrix(np.vstack((np.ones(len(etflist)), -np.eye(len(etflist)))))
                    h = matrix(np.vstack((np.array([[1]]), np.zeros((len(etflist), 1)))))
                    solvers.options['show_progress'] = False
                    solv = solvers.lp(c, G, h)
                    x = np.array(solv['x']).flatten()
                    return x
                maxret_w = solv_maxret(simmeandf, etflist)
                
                # Initiate the linspace of return
                minret = simmeandf@minvar_w
                # minret = simmeandf.min()
                maxret = simmeandf@maxret_w
                # maxret = simmeandf.max()
                retspace = np.linspace(minret, maxret, gridsize)
                
                # Weight calculation
                weightlist = [solv_x(i, simcovdf, simmeandf, etflist) for i in retspace]
                simwdf = [a + b for a, b in zip(simwdf, weightlist)]
                
                iter = iter + 1
            print("Iteration Completed")
            simwdf = [w/Nsim for w in simwdf]
            
            # Normalize
            efstd = [np.sqrt(12 * w@covdf@w) for w in simwdf]
            efret = [12 * w@meandf for w in simwdf]
            SRlist = [sharpe_ratio(w, meandf, covdf, rf) for w in simwdf]
            maxSR = np.argmax(SRlist)
            maxSR_ret = efret[maxSR]
            maxSR_std = efstd[maxSR]
            robw = simwdf[maxSR]
            
            cml_std = np.linspace(0, efstd[-1], gridsize)
            cml_ret = [std * (maxSR_ret - rf*12)/maxSR_std + rf*12 for std in cml_std]
            
            # Report the MV Portfolio Weight
            print("Robust Max Sharpe Ratio Portfolio Weights: ")
            for i in range(len(etflist)): 
                perct = robw[i] * 100
                print(f"Asset {i+1} - {etflist[i]}: {perct.round(2)}%")
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # Create the pie chart
            wedges, texts, autotexts = ax.pie(robw, autopct='%1.1f%%',
                shadow=False, startangle=140)
            ax.legend(wedges, etflist, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=len(etflist))
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            plt.title(f'Robust Max Sharpe Ratio Portfolio Weights, {shortchoice} Short Selling, Date Range: {startdate}-{enddate}')
            plt.show()

            stddf = stddf * np.sqrt(12)
            meandf = meandf * 12 
            
            # Plot
            gl = min(min(efstd), min(stddf)) * 0.7 
            gr = max(max(efstd), max(stddf)) * 1.1 
            gu = max(max(efret), max(meandf)) * 1.15 
            gb = min(min(efret), min(meandf)) * 0.7 
            
            plt.plot(figsize=(15,5))
            plt.plot(efstd, efret, linewidth = 1)
            plt.plot(cml_std, cml_ret, color='red', linewidth = 1)
            plt.scatter(stddf, meandf, color='purple', marker='o', s=40)
            for i in range(len(etflist)): 
                plt.annotate(etflist[i], (stddf[i], meandf[i]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.scatter(maxSR_std, maxSR_ret, color='red', marker='*', s=110)
            plt.text(maxSR_std, maxSR_ret, s="MVP", horizontalalignment='right', verticalalignment='top', fontsize=10)
            plt.gca().set_xlim(left=0)
            plt.gca().set_ylim(bottom=0)
            plt.xlim(gl, gr)
            plt.ylim(gb, gu)
            plt.title(f'Robust MV Portfolio, {shortchoice} Short Selling, Date Range: {startdate}-{enddate}')
            plt.show()

            colors = ['orange', 'blue', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow']
            colorlist = colors[:len(etflist)]
            fig, ax = plt.subplots(figsize=(12, 6))
            bottom = np.zeros_like(efstd) 
            allocations = pd.DataFrame(simwdf, columns = etflist)
            for i, e in enumerate(allocations.columns):
                ax.fill_between(efstd, bottom, bottom + allocations[e], label = e, color=colorlist[i], alpha=0.5)
                bottom += allocations[e]  
            plt.title(f'Robust Efficient Frontier Transition Map, Date Range: {startdate}-{enddate}')
            plt.xlabel('Standard Deviation')
            plt.ylabel('Allocation')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(etflist))
            plt.show()

            print("Robust Efficient Frontier Portfolios:")
            
            efpdf1 = pd.DataFrame(simwdf, columns = etflist)
            efpdf2 = pd.DataFrame({'Return': efret, 'Std': efstd, 'SR': SRlist})
            efpdf = pd.concat([efpdf1, efpdf2], axis=1)
            efpdf = efpdf.round(4)
            efpdf.index = efpdf.index + 1
            efpdf.index.name = '#'
            print(tabulate(efpdf, headers='keys', tablefmt='github'))

            
    except Exception as e:
        import traceback
        traceback.print_exc() 

        print(e)
        # mv(df, etflist, short, 0, normal, startdate, enddate)

def backtesting_aux(start_date, end_date, tickers, allocation, rebalancing, data_short, ff5, start_balance):

    # Start prepping for backtesting
    t = data_short.pivot(index='date', columns='ticker_new', values='ret')
    t_dates = t.index  

    if isinstance(tickers, str):
        tickers = [tickers]
    
    if isinstance(tickers, np.ndarray) and tickers.ndim == 0:
        tickers = [tickers.item()]


    # find the missing ticker
    for ticker in tickers:
        if ticker not in t.columns:
            
            if ticker == "None":
                continue

            # format date (YYYY-MM-DD)
            start_str = datetime.strptime(str(start_date), "%Y%m").strftime("%Y-%m-%d")
            end_str = datetime.strptime(str(end_date), "%Y%m").strftime("%Y-%m-%d")


            try:
                df_yf = yf.download(ticker, start=start_str, end=end_str, interval="1mo", progress=False)

                if df_yf.empty:
                    raise ValueError("yfinance returned empty DataFrame")

                df_yf = df_yf[["Close"]].dropna()
                df_yf["ret"] = df_yf["Close"].pct_change()
                df_yf.dropna(inplace=True)

                df_yf["date"] = df_yf.index.to_period("M").astype(str).str.replace("-", "").astype(int)
                df_yf.set_index("date", inplace=True)

                t[ticker] = df_yf["ret"]

            except Exception as e:
                print(f"❌ Failed to fetch {ticker} from yfinance: {e}")



    t_returns = t[tickers] 

    n_months = len(t_dates)
    n_assets = len(tickers)

    y_aux = np.floor(t_dates / 100).astype(int)  # extract year
    m_aux = (t_dates % 100).astype(int)  # extract month
    d_aux = [calendar.monthrange(y, m)[1] for y, m in zip(y_aux, m_aux)]  # get end of month day
    dates_aux = pd.to_datetime(dict(year=y_aux, month=m_aux, day=d_aux))

    if not n_months == len(ff5):
        print('Error: Number of months for tickers different than rf number of months')
        return

    dollar_amt = np.zeros((n_months, n_assets))
    
    allocation = np.array(allocation, dtype=float)  # Ensure allocation is a numpy array
    
    if rebalancing == 'monthly':
        for t in range(n_months):
            if t == 0:
                dollar_amt[t, :] = start_balance * np.nan_to_num(allocation / 100) * (1 + t_returns.iloc[t].values)
            else:
                dollar_amt[t, :] = np.sum(dollar_amt[t - 1, :]) * np.nan_to_num(allocation / 100)
                dollar_amt[t, :] *= (1 + t_returns.iloc[t].values)

    elif rebalancing == 'None':
        dollar_amt_start = start_balance * np.nan_to_num(allocation / 100)
        cum_returns = (1 + t_returns).cumprod()
        dollar_amt = np.tile(dollar_amt_start, (n_months, 1)) * cum_returns.values

    elif rebalancing == 'yearly':
        for t in range(n_months):
            if t == 0:
                dollar_amt[t, :] = start_balance * np.nan_to_num(allocation / 100) * (1 + t_returns.iloc[t].values)
            elif t % 12 != 0:  # Rebalance yearly 
                dollar_amt[t, :] = dollar_amt[t - 1, :] * (1 + t_returns.iloc[t].values)
            else:
                dollar_amt[t, :] = np.sum(dollar_amt[t - 1, :]) * np.nan_to_num(allocation / 100)
                dollar_amt[t, :] *= (1 + t_returns.iloc[t].values)

    # Portfolio value
    pv = np.sum(dollar_amt, axis=1)
    pv2 = np.concatenate(([start_balance], pv))

    # Annual returns
    ann_return_cagr = (pv2[-1] / start_balance) ** (12 / n_months) - 1
    ann_return_average = np.nanmean(np.diff(pv2) / pv2[:-1]) * 12
    ann_std = np.nanstd(np.diff(pv2) / pv2[:-1]) * np.sqrt(12)
    sharpe_ratio = (ann_return_average - np.nanmean(ff5['RF'])*12) / ann_std
    p_returns = np.diff(pv2) / pv2[:-1]

    num = p_returns - ff5['RF'].values
    den = np.where(num > 0, 0, num)
    sortino_ratio = np.nanmean(num) * 12 / (np.sqrt(12) * np.nanstd(den))

    # Annual returns by year
    unique_years = np.unique(y_aux)
    ann_ret = []
    for y in unique_years:
        returns_aux = p_returns
        indicator = y_aux == y
        returns_y = returns_aux[indicator]
        aux_ret = np.cumprod(returns_y + 1)

        ann_ret.append([y, aux_ret[-1] - 1, len(returns_y), np.min(m_aux[indicator]), np.max(m_aux[indicator])])

    # Compute drawdowns
    cumulative_max = np.maximum.accumulate(pv)
    drawdowns = (pv - cumulative_max) / cumulative_max

    # Group drawdowns
    drawdown_group = np.full_like(drawdowns, np.nan)
    non_zero_mask = drawdowns != 0
    start_of_group = np.concatenate(([False], np.diff(non_zero_mask.astype(int)) > 0))
    group_ids = np.cumsum(start_of_group)
    drawdown_group[non_zero_mask] = group_ids[non_zero_mask]

    # Get worst 3 drawdowns
    min_values = pd.Series(drawdowns[non_zero_mask]).groupby(drawdown_group[non_zero_mask]).min()
    min_values_result = min_values.sort_values().head(3)

    # Prepare drawdown table
    drawdowns_tab = []
    for j, (group_id, min_val) in enumerate(min_values_result.items()):
        indicator = drawdown_group == group_id
        drawdowns_short = drawdowns[indicator]
        dates_short = dates_aux[indicator]
        start_date = dates_short.min()
        end_date = dates_short.iloc[np.argmin(drawdowns_short)]
        no_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        recovery_date = dates_short.max()
        recovery_time = (recovery_date.year - end_date.year) * 12 + (recovery_date.month - end_date.month)
        underwater_period = (recovery_date.year - start_date.year) * 12 + (recovery_date.month - start_date.month)

        drawdowns_tab.append([j + 1, start_date, end_date, no_months, recovery_date, recovery_time, underwater_period, min_val])

    # Format drawdowns table into a DataFrame
    drawdowns_tab2 = pd.DataFrame(drawdowns_tab, columns=[
        'Rank', 'Start date', 'End date', 'Length', 'Recovered by', 'Recovery time', 'Underwater period', 'Drawdown'
    ])
    drawdowns_tab2['Start date'] = drawdowns_tab2['Start date'].dt.strftime('%b-%Y')
    drawdowns_tab2['End date'] = drawdowns_tab2['End date'].dt.strftime('%b-%Y')
    drawdowns_tab2['Recovered by'] = drawdowns_tab2['Recovered by'].dt.strftime('%b-%Y')

    return allocation, tickers, dates_aux, drawdowns, ann_ret, sortino_ratio, sharpe_ratio, pv, ann_return_cagr, ann_return_average, ann_std, p_returns, drawdowns_tab2

def backtesting(start_date, end_date, tickers, allocation1, allocation2, allocation3, rebalancing, benchmark, start_balance):

    # Ensure allocations are properly handled as numeric arrays
    allocation1 = np.array(allocation1, dtype=float)
    allocation2 = np.array(allocation2, dtype=float)
    allocation3 = np.array(allocation3, dtype=float)

    # Ensure NaN handling in allocations
    allocation1 = np.where(allocation1 == None, np.nan, allocation1)
    allocation2 = np.where(allocation2 == None, np.nan, allocation2)
    allocation3 = np.where(allocation3 == None, np.nan, allocation3)
    
    # Load Data
    return_data = pd.read_csv("stocks_mf_ETF_data_final.csv")
    return_data['date'] = return_data['year'] * 100 + return_data['month']
    return_data = return_data.drop(columns=['month', 'year'])

    final_data = return_data.sort_values(by=['ticker_new', 'date'])

    # Cross-check allocations
    ind_alloc = [0, 0, 0, 0]  # This vector stores which of the 3 allocations and benchmark should be used
    portfolio_name = ['Portfolio 1', 'Portfolio 2', 'Portfolio 3', 'Benchmark']

    if not np.isnan(allocation1).all():
        ind_alloc[0] = 1
    if not np.isnan(allocation2).all():
        ind_alloc[1] = 1
    if not np.isnan(allocation3).all():
        ind_alloc[2] = 1
    if benchmark != 'None':
        ind_alloc[3] = 1

    # Some cross-checks
    for p in range(3):
        if ind_alloc[p] == 1:
            if p == 0:
                alloc = allocation1
            elif p == 1:
                alloc = allocation2
            elif p == 2:
                alloc = allocation3

            # Check that each positive weight has a ticker
            indicator = alloc > 0
            n_alloc = sum(indicator)
            tickers_aux = [tickers[i] for i, x in enumerate(indicator) if x]
            n_tickers = sum([1 for t in tickers_aux if t])

            if n_tickers != n_alloc:
                print(f"Error: The number of weights needs to be the same as the number of tickers in {portfolio_name[p]}")
                return

            # Check that every element of allocation is between zero and 100
            indicator = (alloc >= 0) & (alloc <= 100)
            if n_alloc != sum(indicator):
                print(f"Error: Weights need to be between zero and 100 in {portfolio_name[p]}")
                return

            # Check allocation adds up to 100
            if not np.nansum(alloc) == 100:
                print(f"Error: Weights need to add up to 100 in {portfolio_name[p]}")
                return

    # Filter data for the tickers in the allocation
    data_short = final_data[final_data['ticker_new'].isin(tickers)]

    # Remove NaNs in returns
    data_short = data_short.dropna(subset=['ret'])

    # Get the minimum and maximum date for each ticker
    non_empty_idx = [t for t in tickers if t]
    min_date = data_short.groupby('ticker_new')['date'].min()
    max_date = data_short.groupby('ticker_new')['date'].max()

    max_min_date = min_date.max()
    min_max_date = max_date.min()

    if not start_date:
        start_date = max_min_date

    if max_min_date > start_date:
        print(f"Data range will start from {max_min_date} because that is the first available date for ticker {non_empty_idx[min_date.argmax()]}")
        start_date = max_min_date

    if not end_date:
        end_date = min_max_date

    if start_date >= end_date:
        print("Error: Start date cannot be after end date")
        return

    # Filter data based on date range
    data_short = data_short[(data_short['date'] >= start_date) & (data_short['date'] <= end_date)]

    # Get risk-free rate data (Fama-French factors)
    # ff5 = pd.read_csv("F-F_Research_Data_5_Factors_2x3.csv", sep=r'\s+', skiprows=1)
    ff5 = pd.read_csv("F-F_Research_Data_5_Factors_2x3.csv", sep=",", skiprows=1)

    ff5.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    ff5 = ff5[(ff5['date'] >= start_date) & (ff5['date'] <= end_date)]

    # Append benchmark returns to the dataset
    ff5['ret'] = ff5['Mkt-RF'] + ff5['RF']
    benchmark_data = pd.DataFrame({
        'date': ff5['date'],
        'ret': ff5['ret']/100,
        'ticker_new': benchmark
    })
    # benchmark_data = pd.DataFrame({
    #     'date': ff5['date'],
    #     'ret': ff5['ret']/100,
    #     'ticker_new': 'CRSPVW'
    # })
    data_short = pd.concat([data_short, benchmark_data])

    # Keep only risk-free rate for sharpe ratio calculation
    ff5 = ff5[['date', 'RF']]
    ff5['RF'] = ff5['RF']/100
    
    # Process each portfolio allocation
    output = {}
    for p in range(3):
        if ind_alloc[p] == 1:
            if p == 0:
                alloc = allocation1
                tickers2 = [t for i, t in enumerate(tickers) if not np.isnan(alloc[i])]
            elif p == 1:
                alloc = allocation2
                tickers2 = [t for i, t in enumerate(tickers) if not np.isnan(alloc[i])]
            elif p == 2:
                alloc = allocation3
                tickers2 = [t for i, t in enumerate(tickers) if not np.isnan(alloc[i])]

            alloc = [a for a in alloc if not np.isnan(a)]

            results = backtesting_aux(start_date, end_date, tickers2, alloc, rebalancing, data_short, ff5, start_balance)
            output[p] = {key: value for key, value in zip(['allocation', 'tickers', 'dates_aux', 'drawdowns', 'ann_ret', 'sortino_ratio', 'sharpe_ratio', 'pv', 'ann_return_cagr', 'ann_return_average', 'ann_std', 'p_returns', 'drawdowns_tab2'], results)}

    # Benchmark
    p = 3
    alloc = np.array([100])
    tickers2 = np.array(benchmark)
    print(tickers2)
    results = backtesting_aux(start_date, end_date, tickers2, alloc, rebalancing, data_short, ff5, start_balance)
    output[p] = {key: value for key, value in zip(['allocation', 'tickers', 'dates_aux', 'drawdowns', 'ann_ret', 'sortino_ratio', 'sharpe_ratio', 'pv', 'ann_return_cagr', 'ann_return_average', 'ann_std', 'p_returns', 'drawdowns_tab2'], results)}

    # Output results and generate plots
    print(f'Portfolio Backtesting {start_date} - {end_date}')

    for p in range(3):
        if ind_alloc[p] == 1:
            print(f"{portfolio_name[p]} allocation:")
            print(pd.DataFrame(output[p]['allocation'], index=output[p]['tickers'], columns=["Allocation"]))

    # Performance summary
    ps_table = pd.DataFrame(index=[
        'Start Balance', 'End Balance', 'Annualized Return (CAGR)', 'Annualized Standard Deviation', 
        'Best Year', 'Worst Year', 'Maximum drawdown', 'Sharpe Ratio', 'Sortino Ratio', 'Correlation with Benchmark'
    ])
    
    for p in range(4):
        if ind_alloc[p] == 1:
            col_data = [
                start_balance,
                output[p]['pv'][-1],
                output[p]['ann_return_cagr'],
                output[p]['ann_std'],
                max([row[1] for row in output[p]['ann_ret']]),
                min([row[1] for row in output[p]['ann_ret']]),
                min(output[p]['drawdowns']),
                output[p]['sharpe_ratio'],
                output[p]['sortino_ratio'],
                np.corrcoef(output[p]['p_returns'], output[3]['p_returns'])[0, 1]
            ]
            ps_table[portfolio_name[p]] = col_data
    
    print("Performance Summary")
    print(ps_table)

    # Portfolio Growth Plot
    plt.figure(figsize=(12, 4))
    for p in range(4):
        if ind_alloc[p] == 1:
            plt.plot(output[p]['dates_aux'], output[p]['pv'], label=portfolio_name[p], linewidth=2.5)
    plt.legend(loc='upper left')
    plt.title('Portfolio Growth')
    plt.show()

    # Annual Returns Plot
    
    # x values represent the years on the benchmark portfolio
    
    t = next((i for i, x in enumerate(ind_alloc) if x), None)
    x = [row[0] for row in output[t]['ann_ret']]  

    # Filter y_values based on ind_alloc[p] == 1 and only process portfolios that are active
    y_values = np.column_stack([
        [row[1] for row in output[p]['ann_ret']] if ind_alloc[p] == 1 and output[p]['ann_ret'] else [np.nan] * len(x)
        for p in range(4) if ind_alloc[p] == 1
    ])

    # Create a list of portfolio labels for the active portfolios
    active_portfolio_names = [portfolio_name[p] for p in range(4) if ind_alloc[p] == 1]

    # Plotting the annual returns for each active portfolio
    width = 0.2  # Width of each bar
    n_portfolios = y_values.shape[1]  # Number of active portfolios

    plt.figure(figsize=(12, 6))

    # Loop through active portfolios and plot with an offset
    for i in range(n_portfolios):
        plt.bar(np.array(x) + i * width, y_values[:, i] * 100, width, label=active_portfolio_names[i])
    
    plt.xticks(x)
    plt.xlabel('Year')
    plt.ylabel('Annual Returns (%)')
    plt.legend(loc='best')
    plt.title('Annual Returns by Portfolio')
    plt.show()

    # Drawdown Plot
    plt.figure(figsize=(12, 4))
    for p in range(4):
        if ind_alloc[p] == 1:
            plt.plot(output[p]['dates_aux'], output[p]['drawdowns'] * 100, label=portfolio_name[p], linewidth=1.5)
    plt.title('Drawdowns')
    plt.legend()
    plt.show()

    # Get top 3 drawdowns for each portfolio
    for p in range(3):
        if ind_alloc[p] == 1:
            print(f"Top 3 drawdowns {portfolio_name[p]}")
            print(output[p]['drawdowns_tab2']) 


    # Regression analysis
    for p in range(3):
        if ind_alloc[p] == 1:
            print(f"Regression analysis {portfolio_name[p]}")

            # Independent and dependent variables
            x_var = output[p]['p_returns'] - ff5['RF']  # Portfolio returns minus risk-free rate
            y_var = output[3]['p_returns'] - ff5['RF']  # Benchmark returns minus risk-free rate

            # Add a constant to x_var for the intercept in the regression model
            x_var_with_constant = sm.add_constant(x_var)

            # Fit the linear model
            model = sm.OLS(y_var, x_var_with_constant).fit()

            # Print R-squared and Adjusted R-squared
            print(f"R square {model.rsquared:.2f}")
            print(f"Adjusted R square {model.rsquared_adj:.2f}")
            print(f"Observations {int(model.nobs)}")

            # Combine the regression results into a DataFrame for easy display
            aux = pd.DataFrame({
                'Loadings': model.params,
                'Standard Errors': model.bse,
                't-stat': model.tvalues,
                'p-value': model.pvalues
            })
            aux.index = ['alpha', 'beta']
            print(aux)

            # Calculate and print annualized alpha
            annualized_alpha = model.params['const'] * 12 
            print(f'Annualized alpha {annualized_alpha:.4f}')


# %%
# etflist = ['FLCNX','JLGMX','VFTNX','VIIIX','VPMAX','MEIKX','VEIRX',
#            'MEFZX','VEMPX','AMDVX','FLKSX','MVCKX','FOCSX','FCPVX',
#            'RERGX','FISMX','VWILX','VTSNX','VEMIX','VGSNX','FTKFX','VBMPX','PIMIX']

#
# etflist = ['VOO','VXUS','AVUV','AVDV','AVEM']


# %% 
# mv(df)

# %%
# short = 0 (set to zero for not allowing short-sale constraints)
# maxuse = 0 (set to 0 for balanced sample)
# normal = 0 (set to zero for resampling)

# mv(df, etflist, 0, 0, 0, 197012, 202312)

# %%

@app.route('/', methods=['GET'])
def home():
    return 'home'

@app.route('/run', methods=['POST'])
def run_mv():

    try:
        global df

        short = int(request.form.get('short', 0))
        maxuse = int(request.form.get('maxuse', 0))
        normal = int(request.form.get('normal', 0))
        startdate = int(request.form.get('startdate', 197001))
        enddate = int(request.form.get('enddate', 202312))

        etflist_str = request.form.get('etflist')
        etflist = etflist_str.split(',') if etflist_str else ['VOO', 'VXUS', 'AVUV', 'AVDV', 'AVEM']

        df = df[(df["ym"] >= startdate) & (df["ym"] <= enddate)]

        plt.switch_backend('Agg')
        f = io.StringIO()
        with redirect_stdout(f):
            mv(df, etflist, short, maxuse, normal, startdate, enddate)
        output_text = f.getvalue()

        image_paths = []
        for idx, fig_num in enumerate(plt.get_fignums()):
            fig = plt.figure(fig_num)
            path = os.path.join(STATIC_DIR, f"plot_{idx}.png")
            fig.savefig(path)
            image_paths.append(f"/static/plot_{idx}.png")
        plt.close('all')

        return jsonify({
            "output_text": output_text,
            "image_urls": image_paths
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc() 
        return jsonify({
            "output_text": f"error: {str(e)}",
            "image_urls": []
        })


@app.route("/backtest", methods=["POST"])
def run_backtest():
    try:
        data = request.json

        start_date_str = data.get("start_date", "1970-01-01")
        end_date_str = data.get("end_date", "2023-12-31")

        start_date = int(start_date_str.replace("-", "")[:6])
        end_date = int(end_date_str.replace("-", "")[:6])
        tickers = data.get("tickers", [])
        allocation1 = [float(x) if x not in [None, ""] else None for x in data.get("allocation1", [])]
        allocation2 = [float(x) if x not in [None, ""] else None for x in data.get("allocation2", [])]
        allocation3 = [float(x) if x not in [None, ""] else None for x in data.get("allocation3", [])]
        rebalancing = data.get("rebalance", "monthly")
        benchmark = data.get("benchmark", ["CRSPVW"])[0]
        start_balance = float(data.get("start_balance", 10000))

        # Catch print() output
        f = io.StringIO()
        plt.switch_backend("Agg")
        with redirect_stdout(f):
            backtesting(start_date, end_date, tickers, allocation1, allocation2, allocation3, rebalancing, benchmark, start_balance)
        # backtesting(start_date, end_date, tickers, allocation1, allocation2, allocation3, rebalancing, benchmark, start_balance)

        output_text = f.getvalue()

        # save image as static file
        image_paths = []
        for i, fig_num in enumerate(plt.get_fignums()):
            fig = plt.figure(fig_num)
            path = os.path.join(STATIC_DIR, f"backtest_plot_{i}.png")
            fig.savefig(path)
            image_paths.append(f"/static/backtest_plot_{i}.png")
        plt.close("all")

        return jsonify({
            "output_text": output_text,
            "image_urls": image_paths,
            "summary_table": [],
            "regression_table": []
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/static/<path:filename>')
def serve_image(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
