import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeSeries:

    def __init__(self):
        logging.info("trend analyzer")

    def daily_publication(self, df: pd.DataFrame, date_column: str, freq: str = 'D') -> pd.DataFrame:
        """
        Aggregate article counts by time frequency.

        Args:
            df (pd.DataFrame): Input DataFrame with a datetime column.
            date_column (str): Name of the datetime column.
            freq (str): Frequency for resampling ('D', 'W', 'M').

        Returns:
            pd.DataFrame: Aggregated publication counts with datetime index.
        """
        try:
            if date_column not in df.columns:
                raise ValueError(f"Column '{date_column}' not found in DataFrame")

            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=[date_column])
            df.set_index(date_column, inplace=True)

            publication_counts = df.resample(freq).size().reset_index(name='article_count')

            logging.info(f"Aggregated publication counts using frequency '{freq}' with {len(publication_counts)} rows.")
            return publication_counts
        except Exception as e:
            logging.error(f"Error aggregating publication counts: {e}")
            return pd.DataFrame()
    
    