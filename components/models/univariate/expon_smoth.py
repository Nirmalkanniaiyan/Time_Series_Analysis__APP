from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import ParameterGrid
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from components.functions import rmse

def ets_forecast(df):
    # Create a 'ds' column with a uniform date time period
    start_date = datetime.today() - timedelta(days=len(df))
    df['ds'] = pd.date_range(start=start_date, periods=len(df), freq='D')

    # Split the dataframe into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df[0:train_size], df[train_size:len(df)]
    
    # Initialize a dictionary to store the models for each column
    models = {}

    # Define the grid of hyperparameters to search
    hyperparameters = {'trend': ['add', 'mul', None], 'seasonal': ['add', 'mul', None], 'seasonal_periods': [None, 4, 12]}

    # Fit a model for each column in the dataframe
    for column in train.columns:
        if column != 'ds':
            best_score, best_params = float("inf"), None
            # Grid search
            for params in ParameterGrid(hyperparameters):
                try:
                    model = ExponentialSmoothing(train[column], trend=params['trend'], seasonal=params['seasonal'], seasonal_periods=params['seasonal_periods'])
                    model_fit = model.fit()
                    rmse_val = rmse(train[column], model_fit.fittedvalues)
                    if rmse_val < best_score:
                        best_score, best_params = rmse_val, params
                except:
                    continue
            # Refit the model with the best parameters found
            model = ExponentialSmoothing(train[column], trend=best_params['trend'], seasonal=best_params['seasonal'], seasonal_periods=best_params['seasonal_periods'])
            model_fit = model.fit()
            models[column] = model_fit
    
    # Initialize a dictionary to store the forecasts
    forecasts = {}

    # Generate a forecast for each column
    for column in test.columns:
        if column != 'ds':
            forecast = models[column].forecast(steps=len(test))
            forecasts[column] = forecast

    # Calculate the RMSE for each column and return the average
    rmse_values = []
    for column in test.columns:
        if column != 'ds':
            rmse_val = rmse(test[column], forecasts[column])
            rmse_values.append(rmse_val)

    avg_rmse = np.mean(rmse_values)
    return avg_rmse, train, test, forecasts