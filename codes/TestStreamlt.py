import streamlit as st
import yfinance as yf
import pandas as pd

class YahooFinanceData:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None

    def fetch_data(self, start=None, end=None, interval='1d'):
        self.data = yf.download(self.ticker, start=start, end=end, interval=interval)
        return self.data

    def get_latest_price(self):
        if self.data is None:
            self.fetch_data()
        return self.data['Close'].iloc[-1]

    def get_summary(self):
        if self.data is None:
            self.fetch_data()
        summary = {
            'Ticker': self.ticker,
            'Last Close': self.get_latest_price(),
            'Mean Close': self.data['Close'].mean(),
            'Median Close': self.data['Close'].median(),
            'Standard Deviation': self.data['Close'].std(),
            'Total Volume': self.data['Volume'].sum()
        }
        return summary

# Streamlit app
def main():
    st.title("Yahoo Finance Data Viewer")

    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
    interval = st.selectbox("Select Interval", ["1d", "1wk", "1mo"])

    if st.button("Fetch Data"):
        stock_data = YahooFinanceData(ticker)
        data = stock_data.fetch_data(start=start_date, end=end_date, interval=interval)
        
        st.subheader(f"Stock Data for {ticker}")
        st.write(data)

        st.subheader("Summary")
        summary = stock_data.get_summary()
        st.write(summary)

        st.subheader("Closing Price Plot")
        st.line_chart(data['Close'])

if __name__ == "__main__":
    main()
