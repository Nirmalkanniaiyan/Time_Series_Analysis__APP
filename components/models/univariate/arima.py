import pandas as pd
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import plotly_express as px

from statsmodels.tsa.stattools import acf, pacf  # For ACF and PACF calculations
import numpy as np  # For array operations


def arima_model(data) :

    # initial_plot = px.line(data, x=data.index, y=data.iloc[:,0].values, title="Initial Plot of the data")
    # return initial_plot

    result = adfuller(data)
    st.write("ADF Statistic: ", result[0])
    st.write("p-value: %f", result[1])

    acf_plot = plot_acf(data)
    st.pyplot(acf_plot)

    pacf_plot = plot_pacf(data)
    st.pyplot(pacf_plot)

    if 0 == 1 :
        # Calculate ACF and PACF
        acf_values = acf(data)
        pacf_values = pacf(data)

        # Create lags (optional, customize based on your needs)
        lags = range(len(acf_values))

        # Create the ACF plot
        fig_acf = px.line(
            x=lags,
            y=acf_values,
            title="Autocorrelation Function (ACF)",
            labels={"x": "Lag", "y": "Autocorrelation"},
            range_y=(-1.2, 1.2),  # Adjust y-axis range for ACF significance bounds
        )

        # Add horizontal lines for significance bounds (optional)
        fig_acf.add_hline(y=1.96 / np.sqrt(len(data)), line_dash="dash", line_color="gray", annotation_text="Upper Bound")
        fig_acf.add_hline(y=-1.96 / np.sqrt(len(data)), line_dash="dash", line_color="gray", annotation_text="Lower Bound")

        # Create the PACF plot
        fig_pacf = px.line(
            x=lags,
            y=pacf_values,
            title="Partial Autocorrelation Function (PACF)",
            labels={"x": "Lag", "y": "Partial Autocorrelation"},
            range_y=(-1.2, 1.2),  # Adjust y-axis range for PACF significance bounds
        )

        # Add horizontal lines for significance bounds (optional)
        fig_pacf.add_hline(y=1.96 / np.sqrt(len(data)), line_dash="dash", line_color="gray", annotation_text="Upper Bound")
        fig_pacf.add_hline(y=-1.96 / np.sqrt(len(data)), line_dash="dash", line_color="gray", annotation_text="Lower Bound")

        fig_acf.update_layout(margin={"r": 0}, width=700)
        fig_pacf.update_layout(margin={"r": 0}, width=700)

        # fig_acf.show()
        # fig_pacf.show()

        st.plotly_chart(fig_acf)
        st.plotly_chart(fig_pacf)

    model = sm.tsa.arima.ARIMA(data, order=(8,1,5))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    forecast = pd.DataFrame(forecast)
    forecast["timestamp"] = pd.date_range(start=datetime.today().date(), periods=30, freq='D')
    forecast.set_index('timestamp', inplace=True)
    forecast.columns = ["forecast"]
    data.columns = ["actual"]
    final = pd.concat([data, forecast], axis=1)

    plot = px.line(final, x=final.index, y=final.columns, title="Forecast using ARIMA model")

    return plot