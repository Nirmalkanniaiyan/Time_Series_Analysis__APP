import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from arch import arch_model
import streamlit as st

def arch(a) :

    total_rows = len(a)
    test_size_percentage = 0.2
    test_size = int(total_rows * test_size_percentage)

    # Define the number of features
    num_features = a.shape[1]

        # Plot PACF for each feature
    for feature in a.columns:
        plt.figure(figsize=(14, 6))
        sgt.plot_pacf(a[feature], lags=40, alpha=0.05, zero=False, method='ols')
        plt.title(f'PACF of Feature {feature}', size=20)
        plt.xlabel('Lag', size=16)
        plt.ylabel('Partial Autocorrelation', size=16)
        # plt.show()
        st.pyplot(plt)


    # Perform rolling forecasts and plot
    fig, axs = plt.subplots(num_features, 1, figsize=(14, 8 * num_features), sharex=True)

    all_rolling_predictions = {}

    for idx, feature in enumerate(a.columns):
        rolling_predictions = []

        # Rolling forecast loop starting from the last 100 values
        for i in range(test_size - 100, len(a)):
            train = a[feature].iloc[:i]
            model = arch_model(train, vol='ARCH', p=5)  # Adjust p as needed
            model_fit = model.fit(disp='off')
            pred = model_fit.forecast(horizon=1)
            rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

        rolling_predictions = pd.Series(rolling_predictions, index=a.index[test_size-100:])
        all_rolling_predictions[feature] = rolling_predictions
        # Plot the last 100 actual values and the rolling forecasted values
        axs[idx].plot(a.index[-100:], a[feature].iloc[-100:], label='Actual')
        axs[idx].plot(rolling_predictions.index[-100:], rolling_predictions[-100:], label='Rolling Forecast', linestyle='--')
        axs[idx].set_title(f'Actual vs Rolling Forecasted Volatility for Feature {feature}', size=20)
        axs[idx].set_xlabel('Time', size=16)
        axs[idx].set_ylabel('Value', size=16)
        axs[idx].legend()

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)