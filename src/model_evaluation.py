import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import root_mean_squared_error

class Evaluation(ABC):
    """ Abstract base class for model evaluation """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """ Evaluate the model on the provided test data """
        pass

class MASE(Evaluation):
    """ Mean Absolute Scaled Error evaluation implementation """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> None:
        """ Calculate Mean Absolute Scaled Error between true and predicted values """
        try:
            mae_model = np.mean(np.abs(y_true - y_pred))
            mae_naive = np.mean(np.abs(np.diff(y_train)))
            mase = round(mae_model / mae_naive, 3)
            open("plots/metric_mase.txt", "w").write(str(mase))
            logging.info(f"Mean Absolute Scaled Error: {mase}")
            return
        
        except Exception as e:
            logging.error(f"Error occurred during MASE calculation: {e}")
            raise

class RMSE(Evaluation):
    """ Root Mean Squared Error evaluation implementation """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """ Calculate Root Mean Squared Error between true and predicted values """
        try:
            rmse = root_mean_squared_error(y_true, y_pred)
            rmse = round(rmse, 3)
            open("plots/metric_rmse.txt", "w").write(str(rmse))
            logging.info(f"Root Mean Squared Error: {rmse}")
            return
        
        except Exception as e:
            logging.error(f"Error occurred during RMSE calculation: {e}")
            raise
