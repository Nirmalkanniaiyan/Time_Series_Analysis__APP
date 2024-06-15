from statsmodels.tsa.statespace.varmax import VARMAX
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from components.functions import rmse

def multivariate_varmax_forecast(df):
    # Split the dataframe into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df[0:train_size], df[train_size:len(df)]
    
    # Fit the VARMAX model
    model = VARMAX(train, order=(1, 1))
    fitted_model = model.fit(disp=False)

    # Generate a forecast
    forecast = fitted_model.forecast(steps=len(test))

    # Calculate the RMSE for each column and return the average
    rmse_values = []
    for i, column in enumerate(test.columns):
        rmse_val = rmse(test[column], forecast.iloc[:, i])
        rmse_values.append(rmse_val)

    avg_rmse = np.mean(rmse_values)
    return avg_rmse, train, test, forecast
