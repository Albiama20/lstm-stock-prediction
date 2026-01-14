import logging
import pandas as pd
from zenml import step
from src.data_ingestion import IngestData

@step(enable_cache = False)
def ingest_df() -> pd.DataFrame:
    
    """ Ingest data step that reads data from a specified path or online source and returns it as a pandas DataFrame. """
    
    try: 
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df

    except Exception as e:
        logging.error(f"Error in ingest data step: {e}")
        raise e