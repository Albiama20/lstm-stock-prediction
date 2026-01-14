import logging
import numpy as np

class TransformData:
    """ Class to transform data """

    def __init__(self, initial_price: float = None):
        self._initial_price = initial_price

    def set_initial_price(self, initial_price: float) -> None:
        self._initial_price = initial_price

    def log_returns(self, prices: np.ndarray) -> np.ndarray:
        
        try:
            log_prices = np.log(prices)
            log_returns = np.diff(log_prices, axis=0)
            return log_returns
        
        except Exception as e:
            logging.error(f"Error occurred during log returns calculation: {e}")
            raise

    def inverse_log_returns(self, log_returns: np.ndarray) -> np.ndarray:
        
        try:
            log_prices = np.concatenate(([np.array(np.log(self._initial_price))], np.cumsum(log_returns) + np.log(self._initial_price)))
            prices = np.exp(log_prices)
            return prices
        
        except Exception as e:
            logging.error(f"Error occurred during inverse log returns calculation: {e}")
            raise