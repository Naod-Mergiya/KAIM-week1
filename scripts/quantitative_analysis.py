import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator


def compute_sentiment_score(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def plot_sentiment_distribution(news_df):
    sentiment_counts = news_df['sentiment_score_word'].value_counts().sort_index()
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}

    sentiment_counts.plot(kind="bar", figsize=(10, 4), title='Sentiment Analysis',
                          xlabel='Sentiment Categories', ylabel='Number of Articles',
                          color=[colors[category] for category in sentiment_counts.index])
    plt.show()


def plot_publisher_sentiment(news_df, publisher):
    publisher_data = news_df[news_df['publisher'] == publisher]
    sentiment_counts = publisher_data['sentiment_score_word'].value_counts().sort_index()
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}

    sentiment_counts.plot(kind="bar", figsize=(10, 4),
                          title=f'Sentiment Analysis of {publisher}',
                          xlabel='Sentiment Categories', ylabel='Number of Articles',
                          color=[colors[category] for category in sentiment_counts.index])
    plt.show()


def check_missing_values(df_aapl, df_amzn, df_goog, df_meta, df_msft, df_nvda, df_tsla):
    combined_missing = pd.concat([
        df_aapl.isnull().sum(),
        df_goog.isnull().sum(),
        df_amzn.isnull().sum(),
        df_msft.isnull().sum(),
        df_meta.isnull().sum(),
        df_nvda.isnull().sum(),
        df_tsla.isnull().sum()
    ], axis=1)
    return combined_missing


def compute_descriptive_statistics(df_aapl, df_amzn, df_goog, df_meta, df_msft, df_nvda, df_tsla):
    stats = pd.concat([
        df_aapl['Close'].describe().to_frame('AAPL'),
        df_goog['Close'].describe().to_frame('GOOG'),
        df_amzn['Close'].describe().to_frame('AMZN'),
        df_msft['Close'].describe().to_frame('MSFT'),
        df_meta['Close'].describe().to_frame('META'),
        df_nvda['Close'].describe().to_frame('NVDA'),
        df_tsla['Close'].describe().to_frame('TSLA')
    ], axis=1)
    return stats


def plot_closing_prices(df_aapl, df_amzn, df_goog, df_meta, df_msft, df_nvda):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    axs[0, 0].plot(df_aapl['Date'], df_aapl['Close'], color='green', label='AAPL')
    axs[0, 0].legend()

    axs[0, 1].plot(df_amzn['Date'], df_amzn['Close'], label='AMZN')
    axs[0, 1].legend()

    axs[0, 2].plot(df_goog['Date'], df_goog['Close'], color='yellow', label='GOOG')
    axs[0, 2].legend()

    axs[1, 0].plot(df_nvda['Date'], df_nvda['Close'], color='brown', label='NVDA')
    axs[1, 0].legend()

    axs[1, 1].plot(df_msft['Date'], df_msft['Close'], color='purple', label='MSFT')
    axs[1, 1].legend()

    axs[1, 2].plot(df_meta['Date'], df_meta['Close'], color='orange', label='META')
    axs[1, 2].legend()

    for ax in axs.flat:
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
    plt.tight_layout()
    plt.show()




def add_technical_indicators(df):
    # Ensure 'Close' column is float
    df['Close'] = df['Close'].astype(float)

    # Simple Moving Average (SMA)
    sma_indicator = SMAIndicator(close=df['Close'], window=20)
    df['SMA'] = sma_indicator.sma_indicator()

    # Exponential Moving Average (EMA)
    ema_indicator = EMAIndicator(close=df['Close'], window=20)
    df['EMA'] = ema_indicator.ema_indicator()

    # Relative Strength Index (RSI)
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()

    # MACD and Signal Line
    macd_indicator = MACD(close=df['Close'])
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()

    return df



def plot_technical_vs_close(df_aapl, df_amzn, df_goog, df_meta, df_msft, df_nvda, indicator):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    tickers = {
        'AAPL': df_aapl, 'AMZN': df_amzn, 'GOOG': df_goog,
        'NVDA': df_nvda, 'MSFT': df_msft, 'META': df_meta
    }

    for i, (ticker, df) in enumerate(tickers.items()):
        row, col = divmod(i, 3)
        axs[row, col].plot(df['Date'], df['Close'], label='Close')
        axs[row, col].plot(df['Date'], df[indicator], label=indicator, color='red')
        axs[row, col].set_title(ticker)
        axs[row, col].legend()
        axs[row, col].set_xlabel('Date')

    plt.tight_layout()
    plt.show()


def plot_rsi_comparison(df_aapl, df_amzn, df_goog, df_meta, df_msft, df_nvda):
    stocks = {
        "AAPL": df_aapl, "GOOG": df_goog, "AMZN": df_amzn,
        "NVDA": df_nvda, "MSFT": df_msft, "META": df_meta
    }

    fig, axs = plt.subplots(6, 2, figsize=(16, 22))
    for i, (name, df) in enumerate(stocks.items()):
        axs[i][0].plot(df['Date'], df['Close'], label="Close")
        axs[i][0].set_title(f"{name} Stock Price")
        axs[i][0].legend()

        axs[i][1].plot(df['Date'], df['RSI'], color='orange', label="RSI")
        axs[i][1].axhline(70, color='red', linestyle='--')
        axs[i][1].axhline(30, color='green', linestyle='--')
        axs[i][1].set_title(f"{name} RSI")
        axs[i][1].legend()

    plt.tight_layout()
    plt.show()


def plot_macd_comparison(df_aapl, df_amzn, df_goog, df_meta, df_msft, df_nvda):
    stocks = {
        "AAPL": df_aapl, "GOOG": df_goog, "AMZN": df_amzn,
        "NVDA": df_nvda, "MSFT": df_msft, "META": df_meta
    }

    fig, axs = plt.subplots(6, 2, figsize=(16, 22))
    for i, (name, df) in enumerate(stocks.items()):
        axs[i][0].plot(df['Date'], df['Close'], label="Close")
        axs[i][0].set_title(f"{name} Stock Price")
        axs[i][0].legend()

        axs[i][1].plot(df['Date'], df['MACD'], color='orange', label="MACD")
        axs[i][1].plot(df['Date'], df['MACD_Signal'], color='red', label="MACD_Signal")
        axs[i][1].axhline(5, color='r', linestyle="--")
        axs[i][1].axhline(-5, color='g', linestyle="--")
        axs[i][1].set_title(f"{name} MACD")
        axs[i][1].legend()

    plt.tight_layout()
    plt.show()



