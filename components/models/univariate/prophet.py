import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import numpy as np

from components.functions import rmse

def prophet_forecast(df):
    # Create a 'ds' column with a uniform date time period
    start_date = datetime.today() - timedelta(days=len(df))
    df['ds'] = pd.date_range(start=start_date, periods=len(df), freq='D')

    # Split the dataframe into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df[0:train_size], df[train_size:len(df)]
    
    # Initialize a dictionary to store the models for each column
    models = {}

    # Fit a model for each column in the dataframe
    for column in train.columns:
        if column != 'ds':
            m = Prophet()
            m.fit(train[['ds', column]].rename(columns={column: 'y'}))
            models[column] = m
    
    # Initialize a dictionary to store the forecasts
    forecasts = {}

    # Generate a forecast for each column
    for column in test.columns:
        if column != 'ds':
            future = models[column].make_future_dataframe(periods=len(test))
            forecast = models[column].predict(future)
            forecasts[column] = forecast

    # Calculate the RMSE for each column and return the average
    rmse_values = []
    for column in test.columns:
        if column != 'ds':
            rmse_val = rmse(test[column], forecasts[column]['yhat'][-len(test):])
            rmse_values.append(rmse_val)

    avg_rmse = np.mean(rmse_values)
    return avg_rmse, train, test, forecasts