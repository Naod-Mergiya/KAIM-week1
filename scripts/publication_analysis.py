import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import statsmodels.api as sm

def analyze_annual_trends(data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze annual publication trends.

    Parameters:
    - data (pd.DataFrame): The data with a 'date' column.

    Returns:
    - pd.DataFrame: DataFrame for annual article counts.
    """
    # Group by year and count articles
    annual_counts = data.groupby(data['date'].dt.to_period('Y')).size().reset_index(name='no_of_articles')
    annual_counts['date'] = annual_counts['date'].dt.to_timestamp()
    
    return annual_counts

def analyze_quarterly_trends(data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze quarterly publication trends.

    Parameters:
    - data (pd.DataFrame): The data with a 'date' column.

    Returns:
    - pd.DataFrame: DataFrame for quarterly article counts.
    """
    # Group by quarter and count articles
    quarterly_counts = data.groupby(data['date'].dt.to_period('Q')).size().reset_index(name='no_of_articles')
    quarterly_counts['date'] = quarterly_counts['date'].dt.to_timestamp()
    
    return quarterly_counts

def plot_long_term_trends(annual_counts: pd.DataFrame, quarterly_counts: pd.DataFrame) -> None:
    """
    Plot the annual and quarterly trends in article publications.

    Parameters:
    - annual_counts (pd.DataFrame): Annual article counts.
    - quarterly_counts (pd.DataFrame): Quarterly article counts.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(annual_counts['date'], annual_counts['no_of_articles'], marker='o', linestyle='-', color='blue')
    plt.title('Annual Article Publication Trends')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(quarterly_counts['date'], quarterly_counts['no_of_articles'], marker='o', linestyle='-', color='purple')
    plt.title('Quarterly Article Publication Trends')
    plt.xlabel('Quarter')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.show()



def decompose_time_series(data: pd.DataFrame, frequency: int) -> None:
    """
    Decompose the time series into trend, seasonality, and residual components.

    Parameters:
    - data (pd.DataFrame): The data with a 'date' column.
    - frequency (str): The frequency for decomposition ('M' for monthly, 'A' for annual).

    Returns:
    - None: Plots the decomposition.
    """
    data.set_index('date', inplace=True)
    
    decomposed = sm.tsa.seasonal_decompose(data['no_of_articles'], model='additive', period=frequency)
    return decomposed