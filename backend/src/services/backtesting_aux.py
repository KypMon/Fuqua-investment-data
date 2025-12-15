#from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import os
#import io
#from contextlib import redirect_stdout
#from flask_cors import CORS
import yfinance as yf
import calendar
import numpy as np
#import statsmodels.api as sm
from datetime import datetime
#import random
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#from statsmodels.stats.stattools import durbin_watson, jarque_bera
#from scipy.stats import skew, kurtosis
from src.logging.app_logger import AppLogger
from src.services.backtest_input_error import BacktestInputError


class BacktestingAux(object):
    def __init__(self):
        self.logger = AppLogger.get_logger()

    def backtesting_aux(self, start_date, end_date, tickers, allocation, rebalancing, data_short, ff5, start_balance):

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

