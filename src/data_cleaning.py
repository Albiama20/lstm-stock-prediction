import logging
import numpy as np
import pandas as pd
from typing import Tuple, Union
from abc import ABC, abstractmethod
from sklearn.preprocessing import RobustScaler
from src.data_transformation import TransformData

class DataStrategy(ABC):
    """ Abstract class defining strategy for handling data """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[float, float, float]]]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """ Strategy for preprocessing data """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        try:
            data["Date"] = pd.to_datetime(data["Date"])
            df = data.drop(labels=0, axis=0)
            return df

        except Exception as e:
            logging.error(f"Error occurred during data preprocessing: {e}")
            raise

class DataDivideStrategy(DataStrategy):
    """ Strategy for splitting data into train and test sets """

    def handle_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[float, float, float]]:
        
        try:
            stock_prices = data['Close'].values

            transform_data = TransformData()
            returns = transform_data.log_returns(stock_prices)
            training_data_len = int(np.ceil(len(returns) * 0.95))
            training_values = stock_prices[:training_data_len + 1]

            scaler = RobustScaler()
            scaled_train = scaler.fit_transform(returns[:training_data_len].reshape(-1, 1))
            scaled_test = scaler.transform(returns[training_data_len:].reshape(-1, 1))
            scaled_returns = np.concatenate((scaled_train, scaled_test), axis=0)

            lookback = 30
            X_train, y_train = [], returns[lookback:training_data_len, :]

            for i in range(lookback, training_data_len):
                X_train.append(scaled_train[i-lookback:i, 0])

            X_test, y_test = [], returns[training_data_len:, :]

            for i in range(training_data_len, len(scaled_returns)):
                X_test.append(scaled_returns[i-lookback:i, 0])

            X = scaled_returns[-lookback:]
            
            X_train, X_test, X = np.array(X_train), np.array(X_test), np.array(X)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            X = np.reshape(X, (1, X.shape[0], 1))

            dates = data['Date'].values[:training_data_len + 1], data['Date'].values[training_data_len + 1:]
            prices = (stock_prices[lookback], stock_prices[training_data_len], stock_prices[-1])

            return X_train, y_train, X_test, y_test, X, training_values, dates, prices

        except Exception as e:
            logging.error(f"Error occurred during data splitting: {e}")
            raise

class DataCleaning:
    """ Context class to use different data handling strategies """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self._data = data
        self._strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[float, float, float]]]:

        try:
            return self._strategy.handle_data(self._data)
        
        except Exception as e:
            logging.error(f"Error occurred during data handling: {e}")
            raise