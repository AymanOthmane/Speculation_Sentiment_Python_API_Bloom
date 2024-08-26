import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def get_OLS(x,y):
    # modelise la regression de x et y
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    results = model.fit()
    return results

def predictive_analysis_serie(serie):
    # regression des données entre elles avec un lag
    x = serie[0:-1].tolist()
    y = serie[1:].tolist()
    results = get_OLS(x, y)
    print(results.summary())
    return results.params,results.pvalues

def predictive_analysis_serie_and_SSI(serie,ssi):
    # regression des données entre elles avec un lag
    x = pd.DataFrame([serie[0:-1].tolist(),ssi[0:-1]],index=[f'{serie}',"SSI"]).T
    y = serie[1:].tolist()
    results = get_OLS(x, y)
    print(results.summary())
    return results.params,results.pvalues

from itertools import chain


def horizon_analysis(serie,ssi):
    res=list()
    lags = chain(range(1,11),range(15,21,5))
    for i in lags:   
        serie[f"lag_{i}"]= serie.pct_change(i)
    serie=serie.reset_index(drop=False).resample('M', on='Date').sum()
    for i in lags:
        x = ssi.tolist()
        y = serie[f'lag_{i}'].tolist()
        # Modélisation de la régression
        results = get_OLS(x, y)
        print(results.summary())
        res.append((results.params,results.pvalues))
    return res

def bootstrap_bias(data,params,nb) :
    betas = []
    stats = []
    rsquared = []
    for _ in range(nb):
        sample = data.sample(frac=1, replace=True)  # Resample with replacement
        res = get_OLS(sample['SSI'],sample.iloc[:,1])
        betas.append(res.params[1])
        stats.append(res.tvalues[1])
        rsquared.append(res.rsquared_adj)
    bias = np.mean(betas) - params[1]
    pvalue = np.mean(np.abs(stats))
    percentiles = [90, 95, 99]
    r2adj = [np.percentile(rsquared, p) for p in percentiles]
    return bias,pvalue,r2adj

def predictive_analysis_ssi(serie,ssi):
    # regression des données entre elles avec un lag
    x = ssi[:-1].tolist()
    y = serie.shift(-1).dropna()
    results = get_OLS(x, y)
    print(results.summary())
    return results.params,results.pvalues

def control_analysis(serie,ssi,control):
    # regression des données entre elles avec un lag
    x = np.column_stack((ssi, control))
    y = serie.shift(-1).dropna()
    results = get_OLS(x, y)
    print(results.summary())
    return results.params,results.pvalues