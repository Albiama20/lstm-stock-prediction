import logging
import numpy as np
from zenml import step
from src.model_development import LSTMModel

@step(enable_cache = False)
def train_model(X_train: np.ndarray, y_train: np.ndarray, prediction_only: bool) -> bool:
    
    """ Model training step that trains a machine learning model on the provided DataFrame """

    trigger = True

    if not prediction_only:
    
        try: 
            model = LSTMModel()
            trained_model = model.train(X_train, y_train)
            trained_model.save("saved_models/LSTM_model.keras")
            logging.info("Model training step completed successfully.")
            return trigger

        except Exception as e:
            logging.error(f"Error during model training: {e}.")
            raise
    
    return trigger