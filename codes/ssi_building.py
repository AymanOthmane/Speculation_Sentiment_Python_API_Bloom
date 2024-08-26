import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_collection import *
 
def building_SSI_bloom(frequence = "ME"):
    parameters = {"startDate" : "2008-01-01","endDate" : "2023-04-30"}
    parameters["tickers"] = ['SSO US Equity','QLD US Equity','DDM US Equity','QID US Equity','SDS US Equity','DXD US Equity']
    parameters['strFields'] = ['EQY_SH_OUT']
    blp = BLP()
    data = retrieve_data_bloom(parameters,blp)["EQY_SH_OUT"]
    data.index = pd.to_datetime(data.index)
    data.columns = [tick[:3] for tick in parameters['tickers']]
    data1 = data.resample(frequence).last().multiply(1000).pct_change().dropna()
    return compute_SSI(data1),data

def building_SSI_yf(frequence = "ME"):
    data = yf_Data.get_ETF_data()
    data.index = pd.to_datetime(data.index)
    data1 = data.resample(frequence).last().multiply(1000).pct_change().dropna()
    return compute_SSI(data1),data

def building_SSI_csv(frequence = "ME"):
    data = pd.read_csv(f"data\DataShrsOutstdg.csv",index_col=0,header=0)
    data.index = pd.to_datetime(data.index)
    data1 = data.resample(frequence).last().multiply(1000).pct_change().dropna()
    return compute_SSI(data1),data

def compute_SSI(datas) :
    SSI = datas['QLD']
    + datas['SSO']
    + datas['DDM']
    - datas['QID']
    - datas['SDS']
    - datas['DXD']
    return SSI.to_frame(name="SSI")
