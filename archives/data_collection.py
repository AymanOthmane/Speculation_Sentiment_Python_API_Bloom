# import blpapi
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

# ## SET UP
# DATE = blpapi.Name("date")
# ERROR_INFO = blpapi.Name("errorInfo")
# EVENT_TIME = blpapi.Name("EVENT_TIME")
# FIELD_DATA = blpapi.Name("fieldData")
# FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
# FIELD_ID = blpapi.Name("fieldId")
# SECURITY = blpapi.Name("security")
# SECURITY_DATA = blpapi.Name("securityData")

# ## Class BLP
# class BLP():
#     #-----------------------------------------------------------------------------------------------------    
#     def __init__(self):
#         """
#             Improve this
#             BLP object initialization
#             Synchronus event handling
            
#         """
#         # Create Session object
#         self.session = blpapi.Session()
        
        
#         # Exit if can't start the Session
#         if not self.session.start():
#             print("Failed to start session.")
#             return
        
#         # Open & Get RefData Service or exit if impossible
#         if not self.session.openService("//blp/refdata"):
#             print("Failed to open //blp/refdata")
#             return
        
#         self.session.openService('//BLP/refdata')
#         self.refDataSvc = self.session.getService('//BLP/refdata')

#         print('Session open')
    
#     #-----------------------------------------------------------------------------------------------------
    
#     def bdh(self, strSecurity, strFields, startdate, enddate, per='DAILY', perAdj = 'CALENDAR', days = 'NON_TRADING_WEEKDAYS', fill = 'PREVIOUS_VALUE', curr = 'EUR'):
#         """
#             Summary:
#                 HistoricalDataRequest ; 
        
#                 Gets historical data for a set of securities and fields

#             Inputs:
#                 strSecurity: list of str : list of tickers
#                 strFields: list of str : list of fields, must be static fields (e.g. px_last instead of last_price)
#                 startdate: date
#                 enddate
#                 per: periodicitySelection; daily, monthly, quarterly, semiannually or annually
#                 perAdj: periodicityAdjustment: ACTUAL, CALENDAR, FISCAL
#                 curr: string, else default currency is used 
#                 Days: nonTradingDayFillOption : NON_TRADING_WEEKDAYS*, ALL_CALENDAR_DAYS or ACTIVE_DAYS_ONLY
#                 fill: nonTradingDayFillMethod :  PREVIOUS_VALUE, NIL_VALUE
                
#                 Options can be selected these are outlined in “Reference Services and Schemas Guide.”    
            
#             Output:
#                 A list containing as many dataframes as requested fields
#             # Partial response : 6
#             # Response : 5
            
#         """
           
#         #-----------------------------------------------------------------------
#         # Create request
#         #-----------------------------------------------------------------------
        
#         # Create request
#         request = self.refDataSvc.createRequest('HistoricalDataRequest')
        
#         # Put field and securities in list is single value is passed
#         if type(strFields) == str:
#             strFields = [strFields]
            
#         if type(strSecurity) == str:
#             strSecurity = [strSecurity]
    
#         # Append list of securities
#         for strF in strFields:
#             request.append('fields', strF)
    
#         for strS in strSecurity:
#             request.append('securities', strS)
    
#         # Set other parameters
#         request.set('startDate', startdate.strftime('%Y%m%d'))
#         request.set('endDate', enddate.strftime('%Y%m%d'))
#         request.set('periodicitySelection', per)
#         request.set('periodicityAdjustment', perAdj )
#         request.set('currency', curr)
#         request.set('nonTradingDayFillOption', days)
#         request.set('nonTradingDayFillMethod', fill)

#         #-----------------------------------------------------------------------
#         # Send request
#         #-----------------------------------------------------------------------

#         requestID = self.session.sendRequest(request)
#         print("Sending request")
        
#         #-----------------------------------------------------------------------
#         # Receive request
#         #-----------------------------------------------------------------------
#         list_msg = []
#         dict_Security_Fields={}
        
#         # Crear
#         for field in strFields:
#                 globals()['dict_'+field] = {}
        
#         while True:
#             event = self.session.nextEvent()
            
#             # Ignores anything that's not partial or final
#             if (event.eventType() !=blpapi.event.Event.RESPONSE) & (event.eventType() !=blpapi.event.Event.PARTIAL_RESPONSE):
#                 continue
            
#             # Extract the response message
#             for msg in blpapi.event.MessageIterator(event):
#                 list_msg.append(msg)
    
            
#             # Break loop if response is final
#             if event.eventType() == blpapi.event.Event.RESPONSE:
#                 break        
        
#         #-----------------------------------------------------------------------
#         # Exploit data 
#         #-----------------------------------------------------------------------
#         for msg in list_msg:
#             ticker = str(msg.getElement(SECURITY_DATA).getElement(SECURITY).getValue())
            
#             for field in strFields:
#                 globals()['dict_'+field][ticker] = {}
            
#             for field_data in msg.getElement(SECURITY_DATA).getElement(FIELD_DATA):
#                 dat = field_data.getElement(0).getValue()
#                 for i in range(1, (field_data.numElements())):
#                     field_name = str(field_data.getElement(i).name())
#                     try:
#                         globals()['dict_'+field_name][ticker][dat] = field_data.getElement(i).getValueAsFloat()
#                     except:
#                         globals()['dict_'+field_name][ticker][dat] = field_data.getElement(i).getValueAsString()
#             for field in strFields:
#                 dict_Security_Fields[field] = pd.DataFrame.from_dict(globals()['dict_'+field], orient = 'columns')
                               
#         return dict_Security_Fields   
    
#     #-----------------------------------------------------------------------------------------------------    
    
#     def closeSession(self):
#         print("Session closed")
#         self.session.stop()

# ## Fonction pour recuperer la donnée
# def retrieve_data_bloom(parameters:dict,blp:BLP):
#     strFields = parameters['strFields']
#     tickers = parameters['tickers']
#     startDate = parameters['startDate']
#     endDate = parameters['endDate']
#     try:
#         output = blp.bdh(tickers,strFields,startDate,endDate)
#     except AttributeError:
#         print("Erreur : La session n'a pas chargé")
#         output = None 
#     return output

class yf_Data:

    df_ETF_date =  'SSO QLD DDM QID SDS DXD'

    @staticmethod
    def get_ETF_data():
        """
        Get historical daily close data for the leveraged ETFs mentionned in the article
        for the dates mentioned in the article from Yahoo Finance.
        """
        df_ETF_date =  'SSO QLD DDM QID SDS DXD'
        data = yf.download(df_ETF_date, start="2008-01-01", end="2023-04-30")['Close']
        return data

    @staticmethod
    def get_yf_data(symbol,start_date,end_date,feature = "Close"):
        """
        Get historical daily close data for the leveraged ETFs mentionned in the article
        for the dates mentioned in the article from Yahoo Finance.
        """
        data = yf.download(symbol, start=start_date, end=end_date)[feature]
        return data

    @staticmethod
    def get_last_close(symbol, period = "1d"):
        """
        Get the latest closing price for the ticker.
        :param symbol: Yahoo Finance Symbol.
        """
        return yf.download(symbol, period=period)['Close']
    
    @staticmethod
    def get_summary(symbol, period = "1y"):
        """
        Get a summary of the stock's performance.
        """
        summary = {
            'Symbol': symbol,
            'Last Close': yf_Data.get_last_close(symbol),
            'Mean Close': yf_Data.get_last_close(symbol, period).mean(),
            'Median Close': yf_Data.get_last_close(symbol, period).median(),
            'Standard Deviation': yf_Data.get_last_close(symbol, period).std(),
        }
        return pd.DataFrame(summary).transpose()


def retrieve_data_yf(parameters:dict):
    tickers = parameters['tickers']
    startDate = parameters['startDate']
    endDate = parameters['endDate']
    try:
        output = yf_Data.get_yf_data(tickers,startDate,endDate)
    except AttributeError:
        print("Erreur : La session n'a pas chargé")
        output = None 
    return output

