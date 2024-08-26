import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def get_OLS(x,y):
    # modelise la regression de x et y
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    return model

def predictive_analysis_serie(serie):
    # regression des données entre elles avec un lag
    x = serie.dropna()
    y = serie.shift(-1).dropna()
    x,y=index_aligned(x,y)
    results = get_OLS(x, y)
    return results,(x,y)

def predictive_analysis_serie_and_SSI(serie,ssi):
    # regression des données entre elles avec un lag
    scaler = StandardScaler()
    ssi = pd.DataFrame(scaler.fit_transform(ssi),index=ssi.index,columns=ssi.columns)
    x = pd.concat([serie,ssi],axis=1).dropna()
    y = serie.shift(-1).dropna()
    x,y=index_aligned(x,y)
    results = get_OLS(x, y)
    return results,(x,y)

def filter_data(df, start_date, end_date=None):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = df.loc[start_date:end_date]
    return filtered_df

def analyse_horizon(serie,ssi,n):
    scaler = StandardScaler()
    ssi = pd.DataFrame(scaler.fit_transform(ssi),index=ssi.index,columns=ssi.columns)
    serie = pd.concat([serie,ssi],axis=1).dropna()
    nom = serie.columns[0]
    results = {}
    coefficients = []
    errors = []
    for i in range(1, n+1):
        serie[f'{nom}_{i}m'] = serie[nom].rolling(window=i).sum().shift(-i)
        X = sm.add_constant(serie['SSI'])
        Y= serie[f'{nom}_{i}m'].dropna()
        X,Y = index_aligned(X,Y)
        model = sm.OLS(Y, X).fit()
        coefficients.append(model.params['SSI'])
        errors.append(model.bse['SSI'])
        results[f'{nom}_{i}m'] = model
    res = {
    'Period': [f'r_{i+1}m' for i in range(0,n)],
    'Coef (SSI)': [results[f'{nom}_{i+1}m'].summary2().tables[1].loc['SSI', 'Coef.'] for i in range(0,n)],
    't-Statistic': [results[f'{nom}_{i+1}m'].summary2().tables[1].loc['SSI', 't'] for i in range(0,n)],
    'pvalues': [results[f'{nom}_{i+1}m'].pvalues['SSI'] for i in range(0,n)],
    'Adj. R²': [results[f'{nom}_{i+1}m'].rsquared_adj for i in range(0,n)],
    'errors' : [results[f'{nom}_{i+1}m'].bse['SSI'] for i in range(0,n)],
    'N': [results[f'{nom}_{i+1}m'].nobs for i in range(0,n)]
    }
    res = pd.DataFrame(res)
    res.set_index('Period', inplace=True)
    return res

def plot_horizon(res,n):
    plt.figure(figsize=(10, 6))
    plt.errorbar([i+1 for i in range(0,n)], res['Coef (SSI)'], yerr=1.96*res['errors'], fmt='o', capsize=5, label='Coefficients avec IC 95%')
    plt.plot([i+1 for i in range(0,n)], res['Coef (SSI)'], label='Coefficients de régression')
    plt.axhline(0, color='grey', linestyle='--')
    plt.title('S&P 500 - Cumulative Return Predictability')
    plt.xlabel('Horizon (months)')
    plt.ylabel('Cumulative Return Predictability')
    plt.legend()
    plt.grid(True)
    return plt.show()

def parametric_bootstrap(serie,ssi, n_iterations=10000):
    np.random.seed(24)
    nom = serie.columns[0]
    serie = pd.concat([serie,ssi],axis=1).dropna()
    X = sm.add_constant(serie[['SSI']])
    y = serie[[nom]].shift(-1).dropna()
    X,y = index_aligned(X,y)
    model = get_OLS(X,y)
    mu, sigma = 0, model.resid.std()
    bootstrap_betas = []
    bootstrap_pvalues = []
    for _ in range(n_iterations):
        errors = np.random.normal(mu, sigma, size=len(X))
        y_bootstrap = model.fittedvalues + errors
        model_bootstrap = sm.OLS(y_bootstrap, X).fit()
        bootstrap_betas.append(model_bootstrap.params['SSI'])
        bootstrap_pvalues.append(model_bootstrap.pvalues['SSI'])
    beta_bias = np.mean(bootstrap_betas) - model.params['SSI']
    results = pd.DataFrame({
    'Beta Estimé': [model.params['SSI']],
    'Biais Beta': [beta_bias],
    'pvalue': [model.pvalues["SSI"]],
    'R² adj': [model.rsquared_adj]
    })
    return results
    


def predictive_analysis_ssi(serie,ssi,shift):
    # regression des données entre elles avec un lag
    scaler = StandardScaler()
    ssi = pd.DataFrame(scaler.fit_transform(ssi),index=ssi.index,columns=ssi.columns)
    x = ssi
    y = serie.shift(shift).dropna()
    x,y=index_aligned(x,y)
    results = get_OLS(x, y)
    return results,(x,y)

def index_aligned(df1,df2):
    common_index = df1.index.intersection(df2.index)
    df1_aligned = df1.loc[common_index]
    df2_aligned = df2.loc[common_index]
    return df1_aligned,df2_aligned

def control_analysis(serie,ssi,control):
    # regression des données entre elles avec un lag
    scaler = StandardScaler()
    ssi = pd.DataFrame(scaler.fit_transform(ssi),index=ssi.index,columns=ssi.columns)
    scaler1 = StandardScaler()
    control = pd.DataFrame(scaler1.fit_transform(control),index=control.index,columns=control.columns)
    x = pd.concat([ssi, control],axis=1).dropna()
    y = serie.shift(-1).dropna()
    x,y=index_aligned(x,y)
    results = get_OLS(x, y)
    return results,(x,y)