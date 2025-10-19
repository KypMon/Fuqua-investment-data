
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
from data_loader import load_csv


class BacktestInputError(Exception):
    """Raised when user input for the backtest fails validation."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or [message]

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
        raise BacktestInputError('Number of months for tickers different than rf number of months')

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
    return_data = load_csv("stocks_mf_ETF_data_final.csv")
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

    # --- Start of new/modified section for returning structured data ---
    returned_structured_data = {
        "portfolio_allocations": [],
        "performance_summary_table": [],
        "drawdown_tables": [],
        "regression_summary_tables": [],
        "portfolio_growth_plot_data": [],
        "annual_returns_plot_data": {"years": [], "series": []},
        "drawdown_plot_data": [],
        "messages": [],
        "warnings": []
    }
    info_messages = []
    # --- End of new/modified section ---

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
                raise BacktestInputError(
                    f"The number of weights needs to be the same as the number of tickers in {portfolio_name[p]}"
                )

            # Check that every element of allocation is between zero and 100
            indicator = (alloc >= 0) & (alloc <= 100)
            if n_alloc != sum(indicator):
                raise BacktestInputError(
                    f"Weights need to be between zero and 100 in {portfolio_name[p]}"
                )

            # Check allocation adds up to 100
            if not np.isclose(np.nansum(alloc), 100):
                raise BacktestInputError(
                    f"Weights need to add up to 100 in {portfolio_name[p]}"
                )

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
        data_range_message = (
            f"Data range will start from {max_min_date} because that is the first available date for ticker "
            f"{non_empty_idx[min_date.argmax()]}"
        )
        print(data_range_message)
        info_messages.append(data_range_message)
        start_date = max_min_date

    if not end_date:
        end_date = min_max_date

    if start_date >= end_date:
        raise BacktestInputError("Start date cannot be after end date")

    # Filter data based on date range
    data_short = data_short[(data_short['date'] >= start_date) & (data_short['date'] <= end_date)]

    # Get risk-free rate data (Fama-French factors)
    # ff5 = load_csv("F-F_Research_Data_5_Factors_2x3.csv", sep=r'\s+', skiprows=1)
    ff5 = load_csv("F-F_Research_Data_5_Factors_2x3.csv", sep=",", skiprows=1)

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
    results = backtesting_aux(start_date, end_date, tickers2, alloc, rebalancing, data_short, ff5, start_balance)
    output[p] = {key: value for key, value in zip(['allocation', 'tickers', 'dates_aux', 'drawdowns', 'ann_ret', 'sortino_ratio', 'sharpe_ratio', 'pv', 'ann_return_cagr', 'ann_return_average', 'ann_std', 'p_returns', 'drawdowns_tab2'], results)}

    # Output results and generate plots
    summary_message = f'Portfolio Backtesting {start_date} - {end_date}'
    print(summary_message)
    info_messages.append(summary_message)

    # !!!
    # Portfolio allocations
    for p in range(3):  # Iterate through the three main portfolios
        if ind_alloc[p] == 1:
            # Keep print for existing text output
            print(f"{portfolio_name[p]} allocation:")
            # Create DataFrame for printing, ensure 'Allocation' column has a clear name
            df_alloc_print = pd.DataFrame(
                output[p]['allocation'], 
                index=output[p]['tickers'], 
                columns=["Allocation (%)"] 
            )
            print(df_alloc_print.to_string()) # Use to_string() for better console formatting
            # Prepare allocation data for JSON return
            allocation_data_for_json = []
            for ticker_idx, ticker_name in enumerate(output[p]['tickers']):
                alloc_value_raw = output[p]['allocation'][ticker_idx]
                # Ensure values are Python native types for JSON serialization
                alloc_value = None
                if alloc_value_raw is not None and not np.isnan(alloc_value_raw):
                    alloc_value = float(alloc_value_raw)
                
                allocation_data_for_json.append({
                    "ticker": ticker_name,
                    "Allocation": alloc_value  # Frontend might expect "allocation" (lowercase)
                })
            
            returned_structured_data["portfolio_allocations"].append({
                "portfolioName": portfolio_name[p],
                "allocations": allocation_data_for_json
            })

    # 2. Data for Portfolio Growth Plot
    portfolio_growth_series_data = []
    # pv2 (portfolio value including start_balance) is created in backtesting_aux
    # For plotting, we often need the starting point.
    # Assuming dates_aux from backtesting_aux aligns with pv (which is pv2[1:])
    
    # Get the first date for the x-axis correctly
    first_date_str_for_start_balance = None
    if output[0] and 'dates_aux' in output[0] and len(output[0]['dates_aux']) > 0:
         # Create a date for the month before the first date in dates_aux for start_balance
        first_period_date = output[0]['dates_aux'][0]
        if first_period_date.month == 1:
            start_balance_month = 12
            start_balance_year = first_period_date.year - 1
        else:
            start_balance_month = first_period_date.month -1
            start_balance_year = first_period_date.year
        # Get the last day of that month
        start_balance_day = calendar.monthrange(start_balance_year, start_balance_month)[1]
        first_date_str_for_start_balance = datetime(start_balance_year, start_balance_month, start_balance_day).strftime('%Y-%m-%d')

    for p_idx in range(4):
        if ind_alloc[p_idx] == 1 and output[p_idx] and 'dates_aux' in output[p_idx] and 'pv' in output[p_idx]:
            # dates_aux from backtesting_aux corresponds to end-of-month dates for pv values
            date_strings = [date_obj.strftime('%Y-%m-%d') for date_obj in output[p_idx]['dates_aux']]
            # pv from backtesting_aux is the end-of-month portfolio values
            pv_list = [float(val) for val in output[p_idx]['pv']]

            # Prepend the start_balance at the actual start date
            # The 'dates_aux' starts after the first period, 'pv' also.
            # 'pv2' in backtesting_aux was np.concatenate(([start_balance], pv))
            # So, we need a date corresponding to start_balance.
            # The first date in 'dates_aux' is the end of the first period.
            # The start_balance is *before* the first period.
            
            plot_dates = date_strings
            plot_values = pv_list

            if first_date_str_for_start_balance:
                plot_dates = [first_date_str_for_start_balance] + date_strings
                plot_values = [float(start_balance)] + pv_list
            
            portfolio_growth_series_data.append({
                "name": portfolio_name[p_idx],
                "dates": plot_dates,
                "values": plot_values
            })
    returned_structured_data["portfolio_growth_plot_data"] = portfolio_growth_series_data

        # 3. Data for Annual Returns Plot
    annual_returns_plot_collector = {"years": [], "series": []}
    # Determine common set of years across all active portfolios
    all_years_set = set()
    for p_idx in range(4):
        if ind_alloc[p_idx] == 1 and output[p_idx] and 'ann_ret' in output[p_idx]:
            for row in output[p_idx]['ann_ret']:
                all_years_set.add(int(row[0])) # row[0] is year
    
    if all_years_set:
        sorted_years = sorted(list(all_years_set))
        annual_returns_plot_collector["years"] = sorted_years

        for p_idx in range(4):
            if ind_alloc[p_idx] == 1 and output[p_idx] and 'ann_ret' in output[p_idx]:
                returns_by_year = {int(row[0]): float(row[1]) * 100 for row in output[p_idx]['ann_ret']}
                series_returns = [returns_by_year.get(year, None) for year in sorted_years]
                
                annual_returns_plot_collector["series"].append({
                    "name": portfolio_name[p_idx],
                    "returns": series_returns
                })
    returned_structured_data["annual_returns_plot_data"] = annual_returns_plot_collector
    
    # 4. Data for Drawdown Plot
    drawdown_plot_series_data = []
    for p_idx in range(4):
        if ind_alloc[p_idx] == 1 and output[p_idx] and 'dates_aux' in output[p_idx] and 'drawdowns' in output[p_idx]:
            date_strings = [date_obj.strftime('%Y-%m-%d') for date_obj in output[p_idx]['dates_aux']]
            drawdown_values = [float(d) * 100 for d in output[p_idx]['drawdowns']]
            drawdown_plot_series_data.append({
                "name": portfolio_name[p_idx],
                "dates": date_strings,
                "values": drawdown_values
            })
    returned_structured_data["drawdown_plot_data"] = drawdown_plot_series_data

    # Performance summary
    ps_table = pd.DataFrame(index=[
        'Start Balance', 'End Balance', 'Annualized Return (CAGR)', 'Annualized Standard Deviation', 
        'Best Year', 'Worst Year', 'Maximum drawdown', 'Sharpe Ratio', 'Sortino Ratio', 'Correlation with Benchmark'
    ])

    # Performance summary (Example of how to structure it for return)
    ps_table_index = [
        'Start Balance', 'End Balance', 'Annualized Return (CAGR)', 'Annualized Standard Deviation',
        'Best Year', 'Worst Year', 'Maximum drawdown', 'Sharpe Ratio', 'Sortino Ratio', 'Correlation with Benchmark'
    ]
    ps_data_for_json = []

    temp_ps_data = {}
    for p_idx in range(4):
        if ind_alloc[p_idx] == 1:
            temp_ps_data[portfolio_name[p_idx]] = [
                start_balance,
                output[p_idx]['pv'][-1],
                output[p_idx]['ann_return_cagr'],
                output[p_idx]['ann_std'],
                max(row[1] for row in output[p_idx]['ann_ret'] if output[p_idx]['ann_ret']), # check if ann_ret is not empty
                min(row[1] for row in output[p_idx]['ann_ret'] if output[p_idx]['ann_ret']), # check if ann_ret is not empty
                min(output[p_idx]['drawdowns']) if output[p_idx]['drawdowns'].size > 0 else None, # Check if drawdowns is not empty
                output[p_idx]['sharpe_ratio'],
                output[p_idx]['sortino_ratio'],
                np.corrcoef(output[p_idx]['p_returns'], output[3]['p_returns'])[0, 1] if p_idx != 3 and output[3]['p_returns'].size > 0 and output[p_idx]['p_returns'].size > 0 else (1.0 if p_idx == 3 else None)
            ]
    
    # Transpose for "metric per row" structure if desired, or keep as "portfolio per column"
    # For a list of dicts (each dict is a row):
    if temp_ps_data: # Check if there's any data to process
        # Convert to a format that's easier to make into list of dicts
        ps_df_intermediate = pd.DataFrame(temp_ps_data, index=ps_table_index)
        ps_df_intermediate = ps_df_intermediate.reset_index().rename(columns={'index': 'Metric'})
        returned_structured_data["performance_summary_table"] = ps_df_intermediate.to_dict(orient='records')

    
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

    for p in range(4): # Including benchmark
        if ind_alloc[p] == 1 and 'drawdowns_tab2' in output[p] and not output[p]['drawdowns_tab2'].empty:
            print(f"\nTop 3 drawdowns {portfolio_name[p]}") # Stays for output_text
            print(output[p]['drawdowns_tab2'].to_string()) # Stays for output_text
            returned_structured_data["drawdown_tables"].append({
                "portfolioName": portfolio_name[p],
                "data": output[p]['drawdowns_tab2'].to_dict(orient='records')
            })
    
    # !!! Performance Summary
    print("Performance Summary")
    print(ps_table)

     # !!! portfolio growth
    # Portfolio Growth Plot
    # plt.figure(figsize=(12, 4))
    # for p in range(4):
    #     if ind_alloc[p] == 1:
    #         plt.plot(output[p]['dates_aux'], output[p]['pv'], label=portfolio_name[p], linewidth=2.5)
    # plt.legend(loc='upper left')
    # plt.title('Portfolio Growth')
    # plt.show()

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

   # !!!
    # Plotting the annual returns for each active portfolio
    width = 0.2  # Width of each bar
    n_portfolios = y_values.shape[1]  # Number of active portfolios

    # plt.figure(figsize=(12, 6))

    # # Loop through active portfolios and plot with an offset
    # for i in range(n_portfolios):
    #     plt.bar(np.array(x) + i * width, y_values[:, i] * 100, width, label=active_portfolio_names[i])
    
    # plt.xticks(x)
    # plt.xlabel('Year')
    # plt.ylabel('Annual Returns (%)')
    # plt.legend(loc='best')
    # plt.title('Annual Returns by Portfolio')
    # plt.show()

    # !!!
    # Drawdown Plot
    # plt.figure(figsize=(12, 4))
    # for p in range(4):
    #     if ind_alloc[p] == 1:
    #         plt.plot(output[p]['dates_aux'], output[p]['drawdowns'] * 100, label=portfolio_name[p], linewidth=1.5)
    # plt.title('Drawdowns')
    # plt.legend()
    # plt.show()

    # !!!
    # Get top 3 drawdowns for each portfolio
    for p in range(3):
        if ind_alloc[p] == 1:
            print(f"Top 3 drawdowns {portfolio_name[p]}")
            print(output[p]['drawdowns_tab2']) 

    


    # !!!
    # Regression analysis
    # Regression analysis (Example of how to structure it for return)
    for p in range(3): # Typically for portfolios 1-3 vs benchmark
        if ind_alloc[p] == 1 and ind_alloc[3] ==1: # Ensure benchmark (output[3]) exists
            print(f"\nRegression analysis {portfolio_name[p]}") # Stays for output_text
            
            # Ensure p_returns and RF are aligned and valid for subtraction
            portfolio_returns_excess = output[p]['p_returns'] - ff5['RF'].values[:len(output[p]['p_returns'])]
            benchmark_returns_excess = output[3]['p_returns'] - ff5['RF'].values[:len(output[3]['p_returns'])]

            # Ensure same length for regression
            min_len = min(len(portfolio_returns_excess), len(benchmark_returns_excess))
            x_var = portfolio_returns_excess[:min_len]
            y_var = benchmark_returns_excess[:min_len]

            if len(x_var) > 1: # Need at least 2 data points for regression
                # x_var is likely a 1D numpy array (portfolio_returns_excess)
                # sm.add_constant(x_var) will make it a 2D array, typically with the first column as 'const'
                x_var_with_constant = sm.add_constant(x_var, has_constant='add')
                model = sm.OLS(y_var, x_var_with_constant).fit()
                
                # Correctly access the 'const' coefficient for annualized alpha
                annualized_alpha_val = np.nan
                if len(model.params) > 0:
                    annualized_alpha_val = model.params[0] * 12 
                else:
                    print("Warning: model.params is empty, cannot calculate alpha.")

                print(f"R square {model.rsquared:.2f}")
                print(f"Adjusted R square {model.rsquared_adj:.2f}")
                
                # Create the coefficients DataFrame using model.model.exog_names
                # model.model.exog_names should be ['const', name_of_x_var]
                coef_df_factors = ['const', 'beta'] # Or more dynamically: model.model.exog_names
                if hasattr(model.model, 'exog_names') and len(model.model.exog_names) == len(model.params):
                    coef_df_factors = model.model.exog_names
                
                coef_df = pd.DataFrame({
                    'Factor': coef_df_factors,
                    'Loadings': model.params,
                    'Standard Errors': model.bse,
                    't-stat': model.tvalues,
                    'p-value': model.pvalues
                })

                print(coef_df.to_string())
                alpha_print_val = float(annualized_alpha_val) if not np.isnan(annualized_alpha_val) else 'N/A'
                if isinstance(alpha_print_val, float):
                    print(f'Annualized alpha {alpha_print_val:.4f}')
                else:
                    print(f'Annualized alpha {alpha_print_val}')

                returned_structured_data["regression_summary_tables"].append({
                    "portfolioName": portfolio_name[p],
                    "r_squared": round(model.rsquared, 4),
                    "adj_r_squared": round(model.rsquared_adj, 4),
                    "annualized_alpha": float(annualized_alpha_val) if not np.isnan(annualized_alpha_val) else None,
                    "n_observations": int(model.nobs),
                    "coefficients": coef_df.to_dict(orient='records')
                })
            else:
                regression_warning = f"Not enough data points for regression for {portfolio_name[p]}"
                print(regression_warning)
                returned_structured_data["warnings"].append(regression_warning)
                returned_structured_data["regression_summary_tables"].append({
                    "portfolioName": portfolio_name[p],
                    "error": "Not enough data points for regression."
                })

    returned_structured_data["messages"] = info_messages
    return returned_structured_data



