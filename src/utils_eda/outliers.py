import logging
import pandas as pd
from scipy.stats import zscore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Outlier_parser:
    
    def __init__(self):
        logging.info("Outlier_parser.")
    
    def outliers_length(self, df: pd.DataFrame, column: str, z_thresh: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in text length using Z-score.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Name of the column containing text lengths.
            z_thresh (float): Z-score threshold for detecting outliers.

        Returns:
            pd.DataFrame: DataFrame containing the outlier rows.
        """
        try:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

            df = df.copy()
            df['z_score'] = zscore(df[column])
            outliers = df[abs(df['z_score']) > z_thresh]

            logging.info(f"Detected {len(outliers)} outliers in '{column}' using Z-score threshold {z_thresh}")
            return outliers
        except Exception as e:
            logging.error(f"Error detecting length outliers: {e}")
            return pd.DataFrame()

    def duplicate_texts(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Detect duplicate rows based on a text column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column containing text data to check for duplicates.

        Returns:
            pd.DataFrame: DataFrame of duplicated rows.
        """
        try:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

            duplicates = df[df.duplicated(subset=[column], keep=False)].copy()
            logging.info(f"Detected {len(duplicates)} duplicate rows based on column '{column}'")
            return duplicates
        except Exception as e:
            logging.error(f"Error detecting duplicate texts: {e}")
            return pd.DataFrame()