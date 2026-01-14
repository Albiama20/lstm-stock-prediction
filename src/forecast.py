import logging
import numpy as np
from keras.models import load_model

class ForecastData:
    """ Class to visualize prediction results """

    def __init__(self):
        self.lookback = 30
        self.file_path = "saved_models/LSTM_model.keras"

    def new_predictions(self, X: np.ndarray) -> np.ndarray:

        try:
            model = load_model(self.file_path)
            predictions = []

            for _ in range(7):

                Y = model.predict(X)[0, 0]
                predictions.append(Y)
                X = np.concatenate((X[:, -(self.lookback - 1):, :], np.array([[[Y]]])), axis=1)

            predictions = np.array(predictions).reshape(-1, 1)
            logging.info("New predictions generated successfully.")

            return predictions       

        except Exception as e:
            print(f"An error occurred during forecasting: {e}")