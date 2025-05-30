import logging
import pandas as pd
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def headline_lengths(df: pd.DataFrame, column: str = "headline") -> Dict[str, Any]:
    """
    Analyze the length distribution of headlines in the dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame containing the headlines.
        column (str): Name of the column containing headlines.

    Returns:
        Dict[str, Any]: Dictionary containing headline length statistics.
    """
    try:
        logging.info(f"Starting headline length analysis on column '{column}'.")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        df["headline_length"] = df[column].astype(str).apply(len)

        stats = {
            "count": df["headline_length"].count(),
            "mean": df["headline_length"].mean(),
            "std": df["headline_length"].std(),
            "min": df["headline_length"].min(),
            "max": df["headline_length"].max(),
            "quantiles": df["headline_length"].quantile([0.25, 0.5, 0.75]).to_dict()
        }

        logging.info("Headline length analysis complete.")
        return stats

    except Exception as e:
        logging.error(f"Error during headline length analysis: {e}")
        return {}