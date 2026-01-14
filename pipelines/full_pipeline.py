from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.prediction import future_predictions

@pipeline(enable_cache=False)
def full_pipeline(prediction_only: bool = True):
    """ A training pipeline that ingests, cleans, trains, and evaluates a model """
    
    raw_data = ingest_df()
    X_train, y_train, X_test, y_test, X, training_values, dates, prices = clean_df(raw_data)
    trigger = train_model(X_train, y_train, prediction_only)
    trigger2 = evaluate_model(X_test, y_test, training_values, y_train, dates, prices, trigger)
    future_predictions(X, dates, prices, trigger2)