import numpy as np
import pandas as pd
import yfinance as y
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar

from pykalman import KalmanFilter
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from ta import add_all_ta_features

from data_collection import * 


class Trading_Strat:
    def __init__(self, df_SSI, bool_arbitrage = False) -> None:
        self.df_SSI = df_SSI.copy()
        self.bool_arbitrage = bool_arbitrage
        self.df_Signal = df_SSI
        self.plot_Signal = None
        self.df_Backtest = None
        self.strategy = None

    def fn_Kalman_signal(self):
        self.strategy = "Kalman_Filter"
        # Define Kalman Filter for estimating the mean
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

        # Observation 
        obs = self.df_SSI.values

        # Apply the Kalman Filter
        state_means, _ = kf.filter(obs)

        # Add the state means to the dataframe
        self.df_Signal[self.strategy] = state_means

        # Generate deltas between state means and observations
        self.df_Signal['Delta'] = 0
        self.df_Signal['Delta'] = self.df_Signal[self.strategy] - self.df_Signal['SSI']

        # Define Signal threshold
        threshold_buy, threshold_sell = self.df_Signal['Delta'].mean() - self.df_Signal['Delta'].std(), self.df_Signal['Delta'].mean() + self.df_Signal['Delta'].std()

        #Generating Signals
        self.df_Signal['Signal'] = np.where(self.df_Signal['Delta'].values > threshold_sell, -1, np.where(self.df_Signal['Delta'].values < threshold_buy, 1, 0))

    def fn_RandomForest(self):
        self.strategy = "Random_Forest"
        data = self.df_Signal.copy()
        # Create features
        data['Returns'] = data['SSI'].pct_change()
        data['SMA'] = data['SSI'].rolling(window=20).mean()
        data['StdDev'] = data['SSI'].rolling(window=20).std()

        # Add technical indicators as features
        data = add_all_ta_features(data, open="SSI", high="SSI", low="SSI", close="SSI", volume="SSI", fillna=True)

        # Calculate turning points
        data['TurningPoint'] = (np.sign(data['Returns'].shift(1)) != np.sign(data['Returns'])).astype(int)
        data = data.dropna()

        # Define features and target
        features = [col for col in data.columns if col not in ['SSI', 'TurningPoint']]
        X = data[features]
        y = data['TurningPoint']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Predict probabilities
        data['Predicted_TurningPoint'] = best_model.predict(X)
        data['Confidence'] = best_model.predict_proba(X)[:, 1]  # Probability of being a turning point

        # Evaluate the model
        accuracy = accuracy_score(y_test, best_model.predict(X_test))
        print(f'Model accuracy: {accuracy:.2f}')

        data['Signal'] = np.where(
                (data['Predicted_TurningPoint'].values == 1) & (data['SSI'].values > 0), -1,
                np.where(
                    (data['Predicted_TurningPoint'].values == 1) & (data['SSI'].values < 0), 1,
                    0
                )
            )
        
        data_cleaned = data[['SSI','Predicted_TurningPoint','Signal']]
        self.df_Signal = data_cleaned

    def plot_strat_signal(self):
            df_data = self.df_Signal
            plt.figure(figsize=(14, 7))

            # Plot the actual closing prices
            plt.plot(df_data.index, df_data['SSI'], label='SSI', color='blue')
            if "Kalman_Filter" in df_data.columns:
                # Plot the Kalman Filter predictions
                plt.plot(df_data.index, df_data[self.strategy], label=f'{self.strategy.replace("_"," ")} Prediction', color='red')
            #if self.strategy == "Random_Forest":
                # Plot the Kalman Filter predictions
                #plt.plot(df_data.index, df_data['Predicted_TurningPoint'], label=f'{self.strategy.replace("_"," ")} Prediction', color='red')

            # Plot the trading signals
            buy_signals = df_data[df_data['Signal'] == 1]
            sell_signals = df_data[df_data['Signal'] == -1]
            plt.scatter(buy_signals.index, buy_signals['SSI'], marker='^', color='green', label='Buy Signal', alpha=1)
            plt.scatter(sell_signals.index, sell_signals['SSI'], marker='v', color='red', label='Sell Signal', alpha=1)

            plt.title(f'SSO Price, {self.strategy.replace("_"," ")} Prediction, and Trading Signals')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()
            self.plot_Signal = plt.show()
            return plt.show()

    def fn_backtest(self):
        us_bd = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
        self.df_Backtest = pd.DataFrame({
            'SSI' : self.df_Signal.SSI.where(self.df_Signal['Signal'] != 0).dropna(),
            'Signal' : self.df_Signal.Signal.where(self.df_Signal['Signal'] != 0).dropna(),
            'entry date' : self.df_Signal.index.where(self.df_Signal['Signal'] != 0).dropna() + us_bd,
            'entry price' : None,
            'exit date' : None,
            'exit price' : None,
            },index=self.df_Signal.index.where(self.df_Signal['Signal'] != 0).dropna())
        self.df_Backtest['entry price'] = self.df_Backtest['entry date'].apply(lambda date: yf_Data.get_yf_data('QLD', date.strftime('%Y-%m-%d'), (date + us_bd).strftime('%Y-%m-%d'), "Open"))
        self.df_Backtest['exit date'] = self.df_Backtest.index.to_frame()['Dates'].apply(lambda date: Trading_Strat.get_exit_date(self,self.df_Signal,date))
        self.df_Backtest['exit price'] = self.df_Backtest['exit date'].apply(lambda date: yf_Data.get_yf_data('QLD', date.strftime('%Y-%m-%d'), (date + us_bd).strftime('%Y-%m-%d'), "Open"))
        self.df_Backtest['Trade return %'] = np.where(
            self.df_Backtest['Signal'] == 1,
            self.df_Backtest[['entry price', 'exit price']].pct_change(axis=1)['exit price'] * 100,
            np.where(
                self.df_Backtest['Signal'] == -1,
                self.df_Backtest[['exit price', 'entry price']].pct_change(axis=1)['entry price'] * 100,
                np.nan
            )
        )
        self.df_Backtest['Cumulative Return']=self.df_Backtest['Trade return %'].cumsum()
        # self.df_Backtest = df_backtest
        # return df_backtest

    def plot_strat_return(self):

        plt.figure(figsize=(14, 7))

        # Plot the actual closing prices
        plt.plot(self.df_Backtest.index, self.df_Backtest['Cumulative Return'], label='Returns', color='blue')

        plt.title('Stratigy Returns')
        plt.xlabel('Dates')
        plt.ylabel('Returns')
        plt.grid(True)
        plt.legend()
        return plt.show()

    def get_exit_date(self, df, start_date):

        us_bd = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())

        # Convert start_date to datetime if it's not already
        start_date = pd.to_datetime(start_date)

        # Find the row index of the start_date
        if start_date not in df.index:
            raise ValueError(f"Start date {start_date} not found in DataFrame index.")

        # Slice the DataFrame from the start_date onwards
        df_after_start = df.loc[start_date:]
        int_signal_close = df_after_start.loc[start_date, 'Signal'] *(-1)
        # Find the next date where 'Signal' is equal to 0
        next_signal_zero = df_after_start[(df_after_start['Signal'] == 0) | ((df_after_start['Signal'] == int_signal_close))].index

        if len(next_signal_zero) == 0:
            return None  # No such date found

        # Return the first occurrence of 'Signal' equal to 0 after the start_date
        return pd.to_datetime(next_signal_zero[0] + us_bd)