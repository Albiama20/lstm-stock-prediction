import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from zenml import step
from typing import Tuple
from datetime import datetime
from src.forecast import ForecastData
from src.data_transformation import TransformData


@step(enable_cache = False)
def future_predictions(X: np.ndarray, dates: Tuple[np.ndarray, np.ndarray], prices: Tuple[float, float, float], trigger2: bool) -> None:

    """ Step to generate future predictions using the trained model """

    if trigger2:
        
        try: 
            forecast = ForecastData()
            y_new = forecast.new_predictions(X)

            transform_data = TransformData()
            transform_data.set_initial_price(prices[2])
            y_new = transform_data.inverse_log_returns(y_new)
            y_new = y_new.astype(float).ravel()[1:]

            last_date = dates[1][-1].astype('datetime64[ms]').astype(datetime)
            nyse = mcal.get_calendar('NYSE')
            schedule = nyse.schedule(start_date = last_date + pd.Timedelta(days=1), end_date = last_date + pd.Timedelta(days=15))
            next_7_days = schedule.index[:7].to_numpy()
            date_strings = pd.to_datetime(next_7_days).strftime('%Y-%m-%d')
            next_7 = np.full(7, prices[2])

            plt.figure(figsize=(10, 6))
            plt.plot(date_strings, next_7, label = f'Value of the day: {last_date.strftime("%Y-%m-%d")}', color = 'green', linestyle = '--')
            plt.plot(date_strings, y_new, label = 'Predicted', color = 'blue')
            plt.title('Prediction for the next 7 days')
            plt.ylabel('Stock price')
            plt.legend()

            plt.savefig("plots/7_days_prediction.png", dpi=300, bbox_inches="tight")
            plt.close()

        except Exception as e:
            logging.error(f"Error during future predictions: {e}")

    else:
        logging.error(f"Error during prediction.")