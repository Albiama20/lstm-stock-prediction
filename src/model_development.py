import logging
import keras
import numpy as np
from abc import ABC, abstractmethod
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class Model(ABC):
    """ Abstract base class for machine learning models """

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> keras.Model:
        pass

class LSTMModel(Model):
    """ LSTM model implementation """

    def __init__(self):
        pass

    def get_exponential_weights(y_train):
        """ Generate exponential weights for training samples based on half-life """

        n = len(y_train)
        alpha = 1 - np.exp(np.log(0.5) / 125)
        indices = np.arange(n)

        weights = (1 - alpha) ** (n - indices)
        weights /= np.mean(weights)
        
        return weights

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> keras.Model:
        """ Train the LSTM model on the provided training data """

        try:

            logging.info("Starting model training.")
            early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=25, min_lr=0.0001)

            model = keras.models.Sequential()
            model.add(keras.layers.Input(shape=(X_train.shape[1], 1)))

            model.add(keras.layers.LSTM(units=128, return_sequences=True))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.1))

            model.add(keras.layers.LSTM(units=128, return_sequences=False))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.1))

            model.add(keras.layers.Dense(units=64, activation='relu'))
            model.add(keras.layers.Dense(units=32, activation='relu'))
            model.add(keras.layers.Dense(units=1))

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.Huber())

            model.fit(
                X_train, y_train, 
                batch_size = 32,
                epochs = 150,            
                shuffle = False,
                validation_split = 0.1,
                sample_weight = LSTMModel.get_exponential_weights(y_train), 
                callbacks = [early_stopping, reduce_lr]
            )
            logging.info("Model training completed successfully.")
            
            return model

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise