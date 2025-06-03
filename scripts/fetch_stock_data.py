
import pandas as pd
import yfinance as yf
def fetch_historical_data(stocks, start_date="2020-01-01", end_date="2023-01-01"):
    yfinance_data = []
    for stock in stocks:
        data = yf.download(stock, start=start_date, end=end_date)
        data['stock'] = stock
        yfinance_data.append(data)
    return pd.concat(yfinance_data)