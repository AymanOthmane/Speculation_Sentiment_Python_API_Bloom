import numpy as np
import pandas as pd
import yfinance as y
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar

from pykalman import KalmanFilter
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from ta import add_all_ta_features

from data_collection import * 

class Trading_Strat:
    def __init__(self, df_SSI, str_frequency = "ME", bool_arbitrage = False) -> None:
        self.df_SSI = df_SSI
        self.str_frequency = str_frequency
        self.bool_arbitrage = bool_arbitrage
        self.df_Signal = df_SSI
        self.plot_Signal = None
        self.strategy=None
        self.df_Backtest = None

    def fn_Kalman_signal(self):
        self.strategy = "Kalman_Filter"
        # Define Kalman Filter for estimating the mean
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

        # Observation 
        obs = self.df_SSI.values

        # Apply the Kalman Filter
        state_means, _ = kf.filter(obs)

        # Add the state means to the dataframe
        self.df_Signal['Kalman_Filter'] = state_means

        # Generate deltas between state means and observations
        self.df_Signal['Delta'] = 0
        self.df_Signal['Delta'] = self.df_Signal['Kalman_Filter'] - self.df_Signal['SSI']

        # Define Signal threshold
        threshold_buy, threshold_sell = self.df_Signal['Delta'].mean() - self.df_Signal['Delta'].std(), self.df_Signal['Delta'].mean() + self.df_Signal['Delta'].std()

        #Generating Signals
        self.df_Signal['Signal'] = np.where(self.df_Signal['Delta'].values > threshold_sell, -1, np.where(self.df_Signal['Delta'].values < threshold_buy, 1, 0))

    def fn_RandomForest(self):
        np.random.seed(272)
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
        # print(data)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # print(data)
        data = data.dropna()
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

    def fn_GRU(self):

        self.strategy = "GRU"

        data = self.df_SSI.copy()
        data.replace(0, 1e-6, inplace=True)
        data = data.dropna()

        # Feature Engineering
        data['Returns'] = data['SSI'].pct_change()
        data['SMA'] = data['SSI'].rolling(window=20).mean()
        data['StdDev'] = data['SSI'].rolling(window=20).std()

        # Add technical indicators as features
        data = add_all_ta_features(data, open="SSI", high="SSI", low="SSI", close="SSI", volume="SSI", fillna=True)
        data.drop(columns=data.columns[(data == 0).all()], inplace=True)

        # Calculate turning points
        data['TurningPoint'] = (np.sign(data['Returns'].shift(1)) != np.sign(data['Returns'])).astype(int)
        data = data.dropna()

        # Define features and target
        features = [col for col in data.columns if col not in ['SSI', 'TurningPoint']]
        X = data[features]
        y = data['TurningPoint']

        # Scale the data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Prepare the dataset for GRU
        def create_dataset(X, y, time_step=1):
            Xs, ys = [], []
            for i in range(len(X) - time_step):
                Xs.append(X[i:(i + time_step)])
                ys.append(y[i + time_step])
            return np.array(Xs), np.array(ys)

        time_step = 20
        X, y = create_dataset(X_scaled, y.values, time_step)

        # Split the data into training and testing sets
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        print(y_train)

        # Define the GRU model
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=(time_step, X.shape[2])))
        model.add(Dropout(0.2))
        model.add(GRU(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

        # Predict turning points
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Evaluate the model
        def evaluate_model(y_true, y_pred, model_name):
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            print(f'{model_name} - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

        evaluate_model(y_test, y_pred, 'GRU Model')

        # Plot the results
        data['Predicted_TurningPoint'] = data['TurningPoint']
        prediction_start_index = data.index[len(X_train) + time_step]
        data.loc[prediction_start_index:, 'Predicted_TurningPoint'] = y_pred.flatten()

        data['Signal'] = np.where(
                        (data['Predicted_TurningPoint'].values == 1) & (data['SSI'].values > 0), -1,
                        np.where(
                            (data['Predicted_TurningPoint'].values == 1) & (data['SSI'].values < 0), 1,
                            0
                        )
                    )
        data_cleaned = data[['SSI', 'Predicted_TurningPoint', 'Signal']]
        self.df_Signal = data_cleaned

        return data




    def plot_strat_signal(self):
        df_data = self.df_Signal
        plt.figure(figsize=(14, 7))

        # Plot the actual closing prices
        plt.plot(df_data.index, df_data['SSI'], label='SSI', color='blue')
        if self.strategy == "Kalman_Filter":
            # Plot the Kalman Filter predictions
            plt.plot(df_data.index, df_data[self.strategy], label=f'{self.strategy.replace("_"," ")} Prediction', color='red')
        #if self.strategy == "Random_Forest":
            # Plot the Kalman Filter predictions
            #plt.plot(df_data.index, df_data['Predicted_TurningPoint'], label=f'{self.strategy.replace("_"," ")} Prediction', color='red')

        # Plot the trading signals
        buy_signals = df_data[df_data['Signal'] == -1]
        sell_signals = df_data[df_data['Signal'] == 1]
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
        self.df_Backtest['entry price'] = self.df_Backtest['entry date'].apply(lambda date: yf_Data.get_yf_data('SSO', date.strftime('%Y-%m-%d'), (date + us_bd).strftime('%Y-%m-%d'), "Open"))
        self.df_Backtest['exit date'] = self.df_Backtest.index.to_frame()['Dates'].apply(lambda date: Trading_Strat.get_exit_date(self,self.df_Signal,date))
        self.df_Backtest['exit price'] = self.df_Backtest['exit date'].apply(lambda date: yf_Data.get_yf_data('SSO', date.strftime('%Y-%m-%d'), (date + us_bd).strftime('%Y-%m-%d'), "Open"))
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
            print(f'pas de date trouvé pour start {start_date}')
            print(df.iloc[-1].name)
            return df.iloc[-1].name  # No such date found

        # Return the first occurrence of 'Signal' equal to 0 after the start_date
        return pd.to_datetime(next_signal_zero[0] + us_bd)
    
    def get_Stat(self):
        """

        """   
        from dateutil.relativedelta import relativedelta
        Vols = self.fn_get_vols(self.df_Backtest['Trade return %'].values, self.str_frequency)
        dbl_Strat_Lengh = relativedelta(self.df_Backtest.index[-1], self.df_Backtest.index[0]).years
        df_Stats = pd.DataFrame(
            {
        "Overall Performance" : self.df_Backtest.iloc[-1,-1],
        'Annualized Performance' : ((1 + (self.df_Backtest.iloc[-1,-1]/100))**(1/dbl_Strat_Lengh))-1,
        'Daily Volatility' : Vols[0],
        'Monthly Volatility' : Vols[1],
        'Annualized Volatility' : Vols[2],
        'Sharp Ratio' : self.fn_sharpe_ratio(self.df_Backtest['Trade return %'].values),
        'Max Drawdown' : self.fn_max_drawdown(),
        'VaR' : self.fn_var(self.df_Backtest['Trade return %']),
            }, index = [self.strategy]
        )
        df_Stats = df_Stats.round(2)
        return df_Stats.transpose()
    
    def fn_get_vols(self,returns,frequency):
    
        if frequency == "D":
            dbl_daily_vol = np.std(returns)
            dbl_annual_vol = dbl_daily_vol * np.sqrt(252)
            dbl_mthly_vol = dbl_annual_vol / np.sqrt(12)


        elif frequency == "ME" or frequency == 'M':
            dbl_mthly_vol = np.std(returns)
            dbl_annual_vol = dbl_mthly_vol * np.sqrt(12)
            dbl_daily_vol = dbl_mthly_vol / np.sqrt(252)

        elif frequency == "W":
            dbl_annual_vol = np.std(returns) * np.sqrt(52)
            dbl_daily_vol = dbl_mthly_vol / np.sqrt(252)
            dbl_mthly_vol = dbl_mthly_vol / np.sqrt(12)

        return [dbl_daily_vol,dbl_mthly_vol,dbl_annual_vol]


    def fn_sharpe_ratio(self,returns, risk_free_rate=10.5):
        """
        Calculate the Sharpe Ratio for a given series of returns.
        
        Parameters:
        - returns: A pandas Series or a numpy array of returns.
        - risk_free_rate: The risk-free rate, defaults to 0.
        
        Returns:
        - sharpe_ratio: The Sharpe Ratio of the returns.
        """
        # Ensure returns is a pandas Series
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        
        # Calculate the mean and standard deviation of the returns
        mean_return = returns.mean()
        std_dev_return = returns.std()
        
        # Calculate the Sharpe Ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev_return
        
        return sharpe_ratio

    def fn_max_drawdown(self):
        
        """
        Calculate the maximum drawdown for a given series of cumulative returns.
        
        Parameters:
        - cumulative_returns: A pandas Series of cumulative returns.
        
        Returns:
        - max_drawdown: The maximum drawdown.
        """
        roll_max = self.df_Backtest['Cumulative Return'].cummax()
        drawdown = self.df_Backtest['Cumulative Return'] / roll_max - 1.0
        max_drawdown = drawdown.cummin().min()
        return max_drawdown
            

    
    def fn_var(self, returns, confidence_level=0.05):
        # Step 1: Sort the returns in ascending order
        sorted_returns = np.sort(returns/100)
        print(sorted_returns)
        # Step 2: Calculate the index for the given confidence level
        index = int(confidence_level * len(sorted_returns))
        
        # Step 3: VaR is the value at the calculated index
        var_value = sorted_returns[index]

        return var_value
