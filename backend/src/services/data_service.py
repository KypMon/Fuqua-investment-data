import pandas as pd
from data_loader import load_csv
from src.logging.app_logger import AppLogger
from src.services.file_service import FileService

class DataService(object):

    def __init__(self, config):
        self.logger = AppLogger.get_logger()

        file_service = FileService(config)

        etf_file = file_service.get_etf_file_path()
        
        return_data = load_csv(etf_file)
        return_data['date'] = return_data['year'] * 100 + return_data['month']
        return_data.drop(columns=['month', 'year'], inplace=True)

        # Regression
        # mom = load_csv('F-F_Momentum_Factor.csv', sep=',')
        mom_file = file_service.get_mom_file_path()
        mom = load_csv(mom_file)
        mom.columns = ['date', 'MOM']
        mom['MOM'] = mom['MOM'].astype('float64')/100

        # ff5 = load_csv('F-F_Research_Data_5_Factors_2x3.csv', sep=',', skiprows=1)
        ff5_file = file_service.get_ff5_file_path()
        ff5 = load_csv(ff5_file, sep=',', skiprows=1)
        ff5.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        for cols in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
            ff5[cols] = ff5[cols].astype('float64')/100
        # @SDS end

        # Merge factors
        all_factors = pd.merge(mom, ff5, on='date', how='outer').sort_values(by='date')

        # Merge return data with factors
        self.final_data = pd.merge(return_data, all_factors, on='date', how='outer').sort_values(by=['ticker_new', 'date'])


    def get_final_data(self):
        return self.final_data