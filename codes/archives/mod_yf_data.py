import yfinance as yf
import pandas as pd
from datetime import date
from pandas.tseries.holiday import USFederalHolidayCalendar

class yf_Data:

    @staticmethod
    def fn_yf_get_ETF_data():
        """
        Get historical daily close data for the leveraged ETFs 
        (SSO QLD DDM QID SDS DXD) mentionned in the article
        for the dates mentioned in the article from Yahoo Finance.
        """
        df_ETF_date =  'SSO QLD DDM QID SDS DXD'
        return yf.download(df_ETF_date, start="2008-01-01", end="2023-04-30")['Close']

    @staticmethod
    def fn_yf_get_last_close(symbol):
        """
        Get the latest closing price for the ticker.
        :param symbol: Yahoo Finance Symbol.
        """
        return yf.download(symbol, period='1d')['Close']
    
    @staticmethod
    def get_yf_data(symbol,start_date,end_date, field = 'Close'):
        """
        Get historical daily close data for the leveraged ETFs mentionned in the article
        for the dates mentioned in the article from Yahoo Finance.
        """
        us_bd = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
        data = yf.download(symbol, start=start_date, end=end_date)[field]
        if "YFPricesMissingError" in data or data.array.size == 0:
            start_date = (pd.to_datetime(start_date) + us_bd).strftime('%Y-%m-%d')
            end_date = (pd.to_datetime(end_date) + us_bd).strftime('%Y-%m-%d')
            data = yf.download(symbol, start=start_date, end=end_date)['Open']
        return round(float(data.values),2)

    @staticmethod
    def get_summary(symbol, period = "1y"):
        """
        Get a summary of the stock's and performance over a given period.
        :param symbol: Yahoo Finance Symbol.
        :param period: (Optional) periode of study
        """

        summary = {
            'Symbol': symbol,
            'Last Close': yf_Data.get_last_close(symbol),
            'Mean Close': yf_Data.get_last_close(symbol, period).mean(),
            'Median Close': yf_Data.get_last_close(symbol, period).median(),
            'Standard Deviation': yf_Data.get_last_close(symbol, period).std(),
        }
        return pd.DataFrame(summary).transpose()