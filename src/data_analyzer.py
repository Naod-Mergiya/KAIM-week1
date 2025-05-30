import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Summary statistics.
    """
    try:
        stats = df.describe()
        logger.info("Computed summary statistics")
        return stats
    except Exception as e:
        logger.error(f"Error computing summary statistics: {str(e)}")
        raise


def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Timestamp column to datetime and extract hour.
    
    Args:
        df (pd.DataFrame): Input DataFrame with Timestamp column.
        
    Returns:
        pd.DataFrame: DataFrame with converted Timestamp and new Hour column.
    """
    try:
        df['date'] = pd.to_datetime(df['date'])
        logger.info("Converted Timestamp to datetime ")
        return df
    except Exception as e:
        logger.error(f"Error converting timestamp: {str(e)}")
        raise
