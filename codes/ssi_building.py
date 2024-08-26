import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_collection import *
 
def building_SSI(parameters):
    parameters["tickers"] = ['SSO US Equity','QLD US Equity','DDM US Equity','QID US Equity','SDS US Equity','DXD US Equity']
    parameters['strFields'] = ['EQY_SH_OUT']
    blp = BLP()
    data = retrieve_data(parameters,blp)["EQY_SH_OUT"]
    data.index = pd.to_datetime(data.index)
    data.columns = [tick[:3] for tick in parameters['tickers']]
    data = data.resample('M').last().multiply(1000).pct_change().dropna()
    return compute_SSI(data)

def compute_SSI(datas) :
    SSI = datas['QLD']
    + datas['SSO']
    + datas['DDM']
    - datas['QID']
    - datas['SDS']
    - datas['DXD']
    return SSI.to_frame(name="SSI")
