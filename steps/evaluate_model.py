import keras
import logging
import numpy as np
import matplotlib.pyplot as plt
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.model_evaluation import MASE, RMSE
from src.data_transformation import TransformData


@step(enable_cache = False)
def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, training_values: np.ndarray, y_train: np.ndarray, dates: Tuple[np.ndarray, np.ndarray], 
                   prices: Tuple[float, float, float], trigger: bool) -> Annotated[bool, "trigger2"]:
    """ Model evaluation step that evaluates the performance of the trained model on the provided DataFrame. """

    try:
        trigger2 = trigger
        model = keras.models.load_model("saved_models/LSTM_model.keras")
        prediction = model.predict(X_test)

        transform_data = TransformData()
        transform_data.set_initial_price(prices[1])
        y_test = y_test.astype(float).ravel()
        y_test = transform_data.inverse_log_returns(y_test)
        prediction = prediction.astype(float).ravel()
        prediction = transform_data.inverse_log_returns(prediction)

        plt.figure(figsize=(10, 6))
        plt.plot(dates[0], training_values, label='Train', color='blue')
        plt.plot(dates[1], y_test[1:], label='True', color='green')
        plt.plot(dates[1], prediction[1:], label='Predicted', color='red')
        plt.title('Google Stock Price Prediction')
        plt.ylabel('Stock price')
        plt.legend()

        plt.savefig("plots/evaluation.png", dpi=300, bbox_inches="tight")
        plt.close()

        transform_data.set_initial_price(prices[0])
        y_train = transform_data.inverse_log_returns(y_train).ravel()

        mase_evaluator, rmse_evaluator = MASE(), RMSE()
        mase_evaluator.calculate_scores(y_test, prediction, y_train)
        rmse_evaluator.calculate_scores(y_test, prediction)

        return trigger2

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise