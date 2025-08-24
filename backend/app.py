
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
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from scipy.stats import skew, kurtosis

from mv import mv
from backtest import backtesting, backtesting_aux
from data_loader import load_csv


app = Flask(
    __name__,
    static_url_path="/static",
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)
CORS(app, resources={r"/*": {"origins": "*"}})
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


### Load CSV data once using the shared loader
RAW_DF = load_csv("stocks_mf_ETF_data_final.csv")
RAW_DF["ym"] = RAW_DF["year"] * 100 + RAW_DF["month"]


return_data = load_csv("stocks_mf_ETF_data_final.csv")
return_data['date'] = return_data['year'] * 100 + return_data['month']
return_data.drop(columns=['month', 'year'], inplace=True)

# Regression
mom = load_csv('F-F_Momentum_Factor.csv', sep=',')
mom.columns = ['date', 'MOM']
mom['MOM'] = mom['MOM'].astype('float64')/100

ff5 = load_csv('F-F_Research_Data_5_Factors_2x3.csv', sep=',', skiprows=1)
ff5.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
for cols in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
    ff5[cols] = ff5[cols].astype('float64')/100

# %% 
# Merge factors
all_factors = pd.merge(mom, ff5, on='date', how='outer').sort_values(by='date')

# %% 
# Merge return data with factors
final_data = pd.merge(return_data, all_factors, on='date', how='outer').sort_values(by=['ticker_new', 'date'])


# %% 
def get_data(file_name):
    # ETF
    try:
        df = load_csv(file_name)
        df = df.pivot_table(index=['year', 'month'], columns = 'ticker_new', values='ret')
        df.reset_index(inplace=True)

        df.drop(columns={'RF'}, inplace = True)
        return df
    # FF
    except pd.errors.ParserError:
        df = load_csv(file_name, skiprows=3)
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
# Import data from CSV file
global_data = get_and_merge(ff_file, etf_file) 


# Calculate sharpe ratio
def sharpe_ratio(x, meandf, covdf, rf): 
    sp = (x@meandf-rf)/np.sqrt(x.T@covdf@x)
    return sp

def mv123(df, etflist = ['BNDX', 'SPSM', 'SPMD', 'SPLG', 'VWO', 'VEA', 'MUB', 'EMB'], short = 0, maxuse = 1, normal = 1, startdate = 199302, enddate = 202312):

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
            Nsim = 100
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
            # fig, ax = plt.subplots()
            # fig.patch.set_facecolor('white')
            # ax.set_facecolor('white')

            # Create the pie chart
            wedges, texts, autotexts = ax.pie(robw, autopct='%1.1f%%',
                shadow=False, startangle=140)
            ax.legend(wedges, etflist, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=len(etflist))
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            plt.title(f'Robust Max Sharpe Ratio Portfolio Weights, {shortchoice} Short Selling, Date Range: {startdate}-{enddate}')
            # plt.show()

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

# @app.route('/run', methods=['POST'])
# def run_mv():
#     try:
#         short    = int(request.form.get('short', 0))
#         maxuse   = int(request.form.get('maxuse', 0))
#         normal   = int(request.form.get('normal', 0))
#         start    = int(request.form.get('startdate', 197001))
#         end      = int(request.form.get('enddate', 202312))
#         etf_str  = request.form.get('etflist','VOO,VXUS,AVUV,AVDV,AVEM')
#         etflist  = etf_str.split(',')

#         # 保留原有 mv() 打印输出
#         buf = io.StringIO()
#         with redirect_stdout(buf):
#             mv(df.copy(), etflist, short, maxuse, normal, start, end)

#         # 额外生成 JSON 图表数据
#         chart_data = run_mv_core(df.copy(), etflist, short, maxuse, normal, start, end)

#         return jsonify({
#             "output_text": buf.getvalue(),
#             **chart_data
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500




# @app.route('/run', methods=['POST'])
# def run_mv_route():
#     try:
#         short   = int(request.form.get('short', 0))
#         maxuse  = int(request.form.get('maxuse', 0))
#         normal  = int(request.form.get('normal', 0))
#         start   = int(request.form.get('startdate', 197001))
#         end     = int(request.form.get('enddate',   202312))
#         etf_str = request.form.get('etflist', "VOO,VXUS,AVUV,AVDV,AVEM")
#         etflist = etf_str.split(",")

#         # 1) 捕获 mv() 的 stdout
#         buf = io.StringIO()
#         with redirect_stdout(buf):
#             mv(global_data.copy(), etflist, short, maxuse, normal, start, end)
#         output_text = buf.getvalue()

#         # 2) 生成交互图数据
#         chart_data = extract_mv_charts(global_data.copy(), etflist, short, maxuse, normal, start, end)

#         return jsonify({
#             'output_text': output_text,
#             **chart_data
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    
@app.route("/run", methods=["POST"])
def run_mv():
    data = request.json or request.form
    etfl = data.get("etflist", "").split(",") if data.get("etflist") else ["VOO","VXUS","AVUV","AVDV","AVEM"]
    short  = int(data.get("short", 0))
    maxuse = int(data.get("maxuse", 0))
    normal = int(data.get("normal", 1))
    sd = int(data.get("startdate", 199302))
    ed = int(data.get("enddate",   202312))

    result = mv(
        global_data.copy(), etfl,
        short, maxuse, normal,
        sd, ed
    )
    return jsonify(result)
    


# @app.route("/backtest", methods=["POST"])
# def run_backtest():
#     try:
#         data = request.json

#         start_date_str = data.get("start_date", "1970-01-01")
#         end_date_str = data.get("end_date", "2023-12-31")

#         start_date = int(start_date_str.replace("-", "")[:6])
#         end_date = int(end_date_str.replace("-", "")[:6])
#         tickers = data.get("tickers", [])
#         allocation1 = [float(x) if x not in [None, ""] else None for x in data.get("allocation1", [])]
#         allocation2 = [float(x) if x not in [None, ""] else None for x in data.get("allocation2", [])]
#         allocation3 = [float(x) if x not in [None, ""] else None for x in data.get("allocation3", [])]
#         rebalancing = data.get("rebalance", "monthly")
#         benchmark = data.get("benchmark", ["CRSPVW"])[0]
#         start_balance = float(data.get("start_balance", 10000))

#         # Catch print() output
#         f = io.StringIO()
#         plt.switch_backend("Agg")
#         with redirect_stdout(f):
#             backtesting(start_date, end_date, tickers, allocation1, allocation2, allocation3, rebalancing, benchmark, start_balance)
#         # backtesting(start_date, end_date, tickers, allocation1, allocation2, allocation3, rebalancing, benchmark, start_balance)

#         output_text = f.getvalue()

#         # save image as static file
#         image_paths = []
#         for i, fig_num in enumerate(plt.get_fignums()):
#             fig = plt.figure(fig_num)
#             path = os.path.join(STATIC_DIR, f"backtest_plot_{i}.png")
#             fig.savefig(path)
#             image_paths.append(f"/static/backtest_plot_{i}.png")
#         plt.close("all")

#         return jsonify({
#             "output_text": output_text,
#             "image_urls": image_paths,
#             "summary_table": [],
#             "regression_table": []
#         })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

@app.route("/backtest", methods=["POST"])
def run_backtest():
    try:
        data = request.json

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
            structured_results_from_backtesting = backtesting(
                start_date, end_date, tickers,
                allocation1, allocation2, allocation3,
                rebalancing, benchmark, start_balance
            )
        
        output_text = f.getvalue()

        image_urls = []
        timestamp = datetime.now().timestamp()
        for i, fig_num in enumerate(plt.get_fignums()): # If plt.show() was indeed removed, this loop might not find figures.
            fig = plt.figure(fig_num)
            img_filename = f"backtest_plot_{timestamp}_{i}.png"
            path = os.path.join(STATIC_DIR, img_filename)
            fig.savefig(path)
            image_urls.append(f"/static/{img_filename}")
        plt.close("all")

        response_data = {
            "output_text": output_text,
            "image_urls": image_urls, # Can be empty if all plots are now frontend-rendered
            "portfolio_allocations": structured_results_from_backtesting.get("portfolio_allocations", []),
            "summary_table": structured_results_from_backtesting.get("performance_summary_table", []),
            "drawdown_tables": structured_results_from_backtesting.get("drawdown_tables", []),
            "regression_table": structured_results_from_backtesting.get("regression_summary_tables", []),
            "portfolio_growth_plot_data": structured_results_from_backtesting.get("portfolio_growth_plot_data", []),
            "annual_returns_plot_data": structured_results_from_backtesting.get("annual_returns_plot_data", {}),
            "drawdown_plot_data": structured_results_from_backtesting.get("drawdown_plot_data", [])
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
            elif isinstance(data, (np.bool_, np.bool8, bool)): # Added bool here
                return bool(data)
            elif pd.isna(data):
                 return None
            return data

        sanitized_response_data = sanitize_for_json(response_data)
        return jsonify(sanitized_response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# def extract_ols_summary(model, x_var, y_var, ticker):
#     print(model.__dict__)

#     """Extracts and structures OLS regression summary data."""
#     n_obs = int(model.nobs)
#     r_squared = model.rsquared
#     adj_r_squared = model.rsquared_adj
#     alpha_annualized = model.params[0] * 12
#     f_statistic = model.fvalue
#     p_value_f = model.f_pvalue
#     aic = model.aic
#     bic = model.bic
#     df_resid = int(model.df_resid)
#     df_model = int(model.df_model)
#     log_likelihood = model.llf
#     cov_type = model.cov_type

#     # Confidence intervals
#     ci = model.conf_int()
#     ci.columns = ["ci_lower", "ci_upper"]

#     coef_table = model.summary2().tables[1]
#     coef_table = coef_table.reset_index().rename(columns={"index": "factor"})
#     coef_table = coef_table.merge(ci, left_on="factor", right_index=True)

#     coefficients = []
#     for _, row in coef_table.iterrows():
#         coefficients.append({
#             "factor": row["factor"],
#             "coef": round(row["Coef."], 4),
#             "std_err": round(row["Std.Err."], 4),
#             "t": round(row["t"], 4),
#             "p_value": round(row["P>|t|"], 4),
#             "ci_lower": round(row["ci_lower"], 4),
#             "ci_upper": round(row["ci_upper"], 4)
#         })

#     diagnostics = {
#         "omnibus": round(model.omni_normtest.statistic, 4),
#         "durbin_watson": round(sm.stats.durbin_watson(model.resid), 4),
#         "jb": round(model.jarque_bera[0], 4),
#         "prob_jb": round(model.jarque_bera[1], 5),
#         "skew": round(model.jarque_bera[2], 4),
#         "kurtosis": round(model.jarque_bera[3], 4),
#         "cond_no": round(np.linalg.cond(x_var), 2)
#     }

#     result = {
#         "r_squared": round(r_squared, 4),
#         "adj_r_squared": round(adj_r_squared, 4),
#         "alpha_annualized": round(alpha_annualized, 4),
#         "n_obs": n_obs,
#         "f_statistic": round(f_statistic, 4),
#         "p_value_f": round(p_value_f, 5),
#         "aic": round(aic, 4),
#         "bic": round(bic, 4),
#         "df_resid": df_resid,
#         "df_model": df_model,
#         "log_likelihood": round(log_likelihood, 4),
#         "cov_type": cov_type,
#         "coefficients": coefficients,
#         "diagnostics": diagnostics
#     }

#     return result

def convert_numpy(obj):
    """Helper function to convert numpy types to native Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj

def extract_ols_summary(model):
    """Extract and structure OLS summary data in a JSON-serializable format."""

        # 回归残差
    resid = model.resid

    # Durbin-Watson 统计量
    dw_stat = durbin_watson(resid)

    # JB 检验：返回 JB统计量、p值、偏度、峰度
    jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(resid)

    # 其他常见指标
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    n_obs = int(model.nobs)
    annualized_alpha = float(model.params[0]) * 12  # 常数项 * 12

    # 汇总为 dict
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

@app.route("/regression", methods=["POST"])
def run_regression():
    try:
        global final_data # Ensure final_data is accessible if it's a global variable

        data = request.json
        ticker = data.get("ticker")
        start_date_str = data.get("start_date", "1970-01-01") 
        end_date_str = data.get("end_date", "2023-12-31")   
        
        start_date = int(start_date_str.replace("-", "")[:6])
        end_date = int(end_date_str.replace("-", "")[:6])
        
        model_name_req = data.get("model", "CAPM") 
        rolling_period = int(data.get("rolling_period", 36))

        if not ticker:
            return jsonify({"error": "Ticker not provided"}), 400

        data_short = final_data[final_data["ticker_new"] == ticker].copy()

        if data_short.empty:
            return jsonify({"error": f"No data found for ticker: {ticker}"}), 400
            
        max_available_date = data_short["date"].max()
        min_available_date = data_short["date"].min()

        if end_date > max_available_date:
            end_date = max_available_date
        if start_date < min_available_date:
            start_date = min_available_date
        
        if start_date > end_date:
             return jsonify({"error": "Start date cannot be after end date for the selected ticker's available range."}), 400

        data_short = data_short[(data_short["date"] >= start_date) & (data_short["date"] <= end_date)]
        
        if data_short.empty:
            return jsonify({"error": f"No data for ticker {ticker} in the specified date range {start_date} - {end_date}"}), 400

        # Prepare y_var (dependent variable: excess returns)
        # Ensure 'RF' (risk-free rate) is present and numeric
        if 'RF' not in data_short.columns:
            return jsonify({"error": "RF (Risk-Free rate) column missing in data_short."}), 500
        data_short['RF'] = pd.to_numeric(data_short['RF'], errors='coerce')
        data_short['ret'] = pd.to_numeric(data_short['ret'], errors='coerce')
        
        y_var_series = (data_short["ret"] - data_short["RF"]).rename('y_excess_return') # Rename for clarity

        nobs_initial = y_var_series.shape[0]
        if nobs_initial == 0:
            return jsonify({"error": "No observations found for y_var after date filtering."}), 400

        # Select factors based on model_name_req
        factor_columns_map = {
            "CAPM": ["Mkt-RF"],
            "FF3": ["Mkt-RF", "HML", "SMB"],
            "FF4": ["Mkt-RF", "HML", "SMB", "MOM"],
            "FF5": ["Mkt-RF", "HML", "SMB", "CMA", "RMW"]
        }
        if model_name_req not in factor_columns_map:
            return jsonify({"error": "Invalid model selected"}), 400
        
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

        if combined_for_regression.shape[0] < x_var_with_constant_df.shape[1] + 1:
            return jsonify({"error": "Not enough data points for regression after handling NaNs in combined y and X."}), 400

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
                
                if not x_roll_df.empty and not y_roll_s.empty and len(y_roll_s) >= x_roll_df.shape[1]:
                    try:
                        mdl_roll = sm.OLS(y_roll_s, x_roll_df, missing='drop').fit()
                        out_roll[k_loop_idx - (rolling_period - 1), :] = mdl_roll.params.values
                    except Exception as e_roll:
                        print(f"Error in rolling regression for window ending at index {k_loop_idx}: {e_roll}")
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
            "rolling_plot_data": rolling_plot_data_for_json 
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
            elif isinstance(data_to_sanitize, (np.bool_, np.bool8, bool)):
                return bool(data_to_sanitize)
            elif pd.isna(data_to_sanitize): 
                 return None
            return data_to_sanitize

        return jsonify(sanitize_for_json(response_payload))

    except Exception as e:
        import traceback
        current_traceback = traceback.format_exc() # Capture traceback string
        print(current_traceback) # Print to server logs
        return jsonify({"error": str(e), "trace": current_traceback}), 500



# @app.route("/regression", methods=["POST"])
# def run_regression():
#     try:
#         global final_data

#         data = request.json
#         ticker = data.get("ticker")
#         start_date = int(data.get("start_date").replace("-", "")[:6])
#         end_date = int(data.get("end_date").replace("-", "")[:6])
#         model = data.get("model", "CAPM")
#         rolling_period = int(data.get("rolling_period", 36))

#         data_short = final_data[final_data["ticker_new"] == ticker]

#         # 日期边界修正
#         if (end_date is None) or (end_date > data_short["date"].max()):
#             end_date = data_short["date"].max()
#         if (start_date is None) or (start_date < data_short["date"].min()):
#             start_date = data_short["date"].min()

#         data_short = data_short[(data_short["date"] >= start_date) & (data_short["date"] <= end_date)]

#         y_var = data_short["ret"] - data_short["RF"]
#         nobs = y_var.shape[0]

#         # 模型选择
#         if model == "CAPM":
#             x_var = data_short[["Mkt-RF"]]
#             factor_names = ["Mkt-Rf"]
#         elif model == "FF3":
#             x_var = data_short[["Mkt-RF", "HML", "SMB"]]
#             factor_names = ["Mkt-Rf", "HML", "SMB"]
#         elif model == "FF4":
#             x_var = data_short[["Mkt-RF", "HML", "SMB", "MOM"]]
#             factor_names = ["Mkt-Rf", "HML", "SMB", "MOM"]
#         elif model == "FF5":
#             x_var = data_short[["Mkt-RF", "HML", "SMB", "CMA", "RMW"]]
#             factor_names = ["Mkt-Rf", "HML", "SMB", "CMA", "RMW"]
#         else:
#             return jsonify({"error": "Invalid model selected"}), 400

#         n_factors = len(factor_names)
#         x_var = sm.add_constant(x_var)
#         mdl = sm.OLS(y_var, x_var, missing='drop').fit()

#         # 回归指标
#         loadings = mdl.params.values
#         se = mdl.bse.values
#         tStat = mdl.tvalues.values
#         pvalue = mdl.pvalues.values
#         rsq = mdl.rsquared
#         adj_rsq = mdl.rsquared_adj
#         alpha_annualized = loadings[0] * 12
#         regression_text = mdl.summary().as_html()

#         print(mdl.summary())

#         # return contribution
#         return_contribution = np.full((n_factors + 2, 2), np.nan)
#         return_contribution[0, 0] = np.nanmean(y_var) * 12
#         return_contribution[1, 0] = alpha_annualized

#         for k in range(n_factors):
#             return_contribution[k + 2, 0] = np.nanmean(x_var.iloc[:, k + 1]) * 12

#         return_contribution[1, 1] = return_contribution[1, 0] / return_contribution[0, 0] * 100
#         return_contribution[2:, 1] = loadings[1:] * np.nanmean(x_var.iloc[:, 1:], axis=0) * 12 / return_contribution[0, 0] * 100

#         return_contribution_df = pd.DataFrame(
#             return_contribution,
#             columns=["Av. Ann. Excess Return", "Return Contribution"],
#             index=[ticker, "alpha"] + factor_names
#         ).round(4).replace({np.nan: None}).reset_index().rename(columns={"index": "Factor"})

#         # 画图
#         image_urls = []
#         if nobs >= rolling_period + 10:
#             out_roll = np.full((nobs - rolling_period + 1, n_factors + 1), np.nan)
#             for k in range(rolling_period, nobs + 1):
#                 x_roll = x_var.iloc[k - rolling_period:k]
#                 y_roll = y_var.iloc[k - rolling_period:k]
#                 mdl_roll = sm.OLS(y_roll, x_roll, missing='drop').fit()
#                 out_roll[k - rolling_period, :] = mdl_roll.params.values

#             date_aux = data_short["date"].iloc[rolling_period - 1:].astype(str)
#             dates_aux = pd.to_datetime(date_aux, format="%Y%m")

#             fig, ax1 = plt.subplots(figsize=(14, 7))
#             ax1.set_xlabel("Date")
#             ax1.set_ylabel("Annualized Alpha", color="tab:red")
#             ax1.plot(dates_aux, out_roll[:, 0] * 12, color="tab:red")
#             ax1.tick_params(axis="y", labelcolor="tab:red")

#             linestyles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
#             ax2 = ax1.twinx()
#             ax2.set_ylabel("Factor Loadings", color="tab:blue")
#             for i in range(n_factors):
#                 ax2.plot(dates_aux, out_roll[:, i + 1], label=factor_names[i], linestyle=linestyles[i])
#             ax2.tick_params(axis="y", labelcolor="tab:blue")
#             fig.tight_layout()
#             fig.legend(["Alpha"] + factor_names, loc="lower right", ncol=n_factors + 1)

#             plot_path = os.path.join(STATIC_DIR, "regression_plot.png")
#             fig.savefig(plot_path)
#             plt.close(fig)
#             image_urls.append("/static/regression_plot.png")

#         return jsonify({
#             "summary_table": return_contribution_df.to_dict(orient="records"),
#             "image_urls": image_urls,
#             "regression_output": {
#                 "r_squared": round(rsq, 4),
#                 "adj_r_squared": round(adj_rsq, 4),
#                 "alpha_annualized": round(alpha_annualized, 4),
#                 "n_observations": int(nobs),
#                 "se": se.tolist(),
#                 "t-stat": tStat.tolist(),
#                 "p-value": pvalue.tolist(),
#                 "text_summary": regression_text
#             }
#         })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_image(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
