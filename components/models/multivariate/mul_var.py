from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import streamlit as st

from components.functions import rmse

def multivariate_var_forecast(df):

    # Split the dataframe into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df[0:train_size], df[train_size:len(df)]
    
    # Fit the VAR model
    model = VAR(train)
    fitted_model = model.fit()

    # Generate a forecast
    forecast = fitted_model.forecast(train.values, steps=len(test))

    # Calculate the RMSE for each column and return the average
    rmse_values = []
    for i, column in enumerate(test.columns):
        rmse_val = rmse(test[column], forecast[:, i])
        rmse_values.append(rmse_val)

    avg_rmse = np.mean(rmse_values)
    return avg_rmse, train, test, forecast
