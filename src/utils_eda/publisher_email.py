import logging
import pandas as pd
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class email_list:
    def __init__(self):
        logging.info("email list")

    def extract_emails(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Extract rows with email addresses in the specified column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column to check for email addresses.

        Returns:
            pd.DataFrame: DataFrame with 'publisher_email' and 'count' columns.
        """
        try:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

            email_pattern = r"[^@\s]+@[^@\s]+\.[^@\s]+"
            email_df = df[df[column].str.contains(email_pattern, na=False, regex=True)].copy()
            email_df['publisher_email'] = email_df[column].str.extract(f"({email_pattern})")

            result = email_df.groupby('publisher_email').size().reset_index(name='count')
            result = result.sort_values(by='count', ascending=False).reset_index(drop=True)

            logging.info(f"Extracted {len(result)} unique publisher email addresses.")
            return result
        except Exception as e:
            logging.error(f"Error extracting email publishers: {e}")
            return pd.DataFrame()