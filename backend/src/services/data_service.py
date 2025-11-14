import pandas as pd
from functools import lru_cache
from typing import Tuple
from src.config.config import Config
from src.logging.app_logger import AppLogger
from src.services.file_service import FileService

class DataService(object):

    def __init__(self):
        self.logger = AppLogger.get_logger()
        self.file_service = FileService()

    def get_final_data(self):
        etf_file = self.file_service.get_etf_file_path()
        
        return_data = self.load_csv(etf_file)
        return_data['date'] = return_data['year'] * 100 + return_data['month']
        return_data.drop(columns=['month', 'year'], inplace=True)

        # Regression
        # mom = load_csv('F-F_Momentum_Factor.csv', sep=',')
        mom_file = self.file_service.get_mom_file_path()
        mom = self.load_csv(mom_file)
        mom.columns = ['date', 'MOM']
        mom['MOM'] = mom['MOM'].astype('float64')/100

        # ff5 = load_csv('F-F_Research_Data_5_Factors_2x3.csv', sep=',', skiprows=1)
        ff5_file = self.file_service.get_ff5_file_path()
        ff5 = self.load_csv(ff5_file, sep=',', skiprows=1)
        ff5.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        for cols in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
            ff5[cols] = ff5[cols].astype('float64')/100
        # @SDS end

        # Merge factors
        all_factors = pd.merge(mom, ff5, on='date', how='outer').sort_values(by='date')

        # Merge return data with factors
        final_data = pd.merge(return_data, all_factors, on='date', how='outer').sort_values(by=['ticker_new', 'date'])
        return final_data
    
    def _make_key(self, filename: str, kwargs: dict) -> Tuple[str, Tuple[Tuple[str, object], ...]]:
        """Create a hashable cache key from filename and keyword arguments."""
        return filename, tuple(sorted(kwargs.items()))

    @lru_cache(maxsize=None)
    def _read_csv_cached(self, filename: str, kwargs_tuple: Tuple[Tuple[str, object], ...]) -> pd.DataFrame:
        """Read a CSV file and cache the resulting DataFrame."""
        kwargs = dict(kwargs_tuple)
        # return pd.read_csv(DATA_DIR / filename, **kwargs)
        return pd.read_csv(filename, **kwargs)


    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        self.logger.info("Loading csv file: " + filename)
        """Load a CSV file from the backend directory with caching.

        Parameters
        ----------
        filename: str
            Name of the CSV file located within the backend directory.
        **kwargs:
            Additional keyword arguments passed to ``pandas.read_csv``.

        Returns
        -------
        pandas.DataFrame
            A copy of the cached DataFrame.
        """
        df = self._read_csv_cached(filename, self._make_key(filename, kwargs)[1])
        return df.copy()

