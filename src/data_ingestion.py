import logging
import pandas as pd
import yfinance as yf

class IngestData:
    """ Class to ingest data from a specified file path or online source."""

    def __init__(self):
        pass
    
    def get_data(self) -> pd.DataFrame:

        try:
            logging.info(f"Reading data from yfinance")
            df = yf.download("GOOG", period="3y", auto_adjust=True)
            df.reset_index(inplace=True)

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise

        return df
