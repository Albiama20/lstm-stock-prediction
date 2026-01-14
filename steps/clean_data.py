import logging
import numpy as np
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataDivideStrategy


@step(enable_cache = False, enable_artifact_metadata = False)
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, "X_train"], 
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_test"],
    Annotated[np.ndarray, "X"],
    Annotated[np.ndarray, "training_values"],
    Annotated[Tuple[np.ndarray, np.ndarray], "dates"],
    Annotated[Tuple[float, float, float], "prices"]
    ]:
    
    """ Data cleaning step that processes the input DataFrame and returns a cleaned DataFrame. """
    
    try:
        process_strategy = DataPreProcessStrategy()
        cleaned_data = DataCleaning(df, process_strategy).handle_data()

        divide_strategy = DataDivideStrategy()
        X_train, y_train, X_test, y_test, X, training_values, dates, prices = DataCleaning(cleaned_data, divide_strategy).handle_data()

        return X_train, y_train, X_test, y_test, X, training_values, dates, prices
    
    except Exception as e:
        logging.error(f"Error in data cleaning step: {e}")
        raise