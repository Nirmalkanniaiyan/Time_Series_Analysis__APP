import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import plotly_express as px
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

def stl_model(f1) :

    stl = STL(f1,period=5)
    res = stl.fit()
    fig = res.plot()

    stlf = STLForecast(f1[:975], ARIMA, model_kwargs=dict(order=(3, 1, 3), trend="t"),period=14)
    stlf_res = stlf.fit()

    forecast = stlf_res.forecast(25)

    # plt.plot(f1[950:1000])

    # plt.plot(forecast)
    # plt.show()

    final = pd.concat([f1, forecast], axis=1)

    plot1 = px.line(final, x=final.index, y=final.columns, title="Forecast using STL model")

    model = sm.tsa.arima.ARIMA(f1[:975], order=(3, 1, 3))
    model_fit = model.fit()

    forecast = model_fit.forecast(25)

    plt.plot(f1[950:1000])

    plt.plot(forecast)
    plt.show()

    final = pd.concat([f1, forecast], axis=1)

    plot2 = px.line(final, x=final.index, y=final.columns, title="Forecast using STL model")

    return (plot1,plot2,fig)