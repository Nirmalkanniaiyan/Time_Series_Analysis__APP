from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly_express as px
from components.functions import get_node_data_from_merged, get_feature_with_dates
from components.models.univariate.arima import arima_model
from components.models.stl import stl_model
import numpy as np

from components.models.univariate.prophet import prophet_forecast
from components.models.univariate.expon_smoth import ets_forecast
from components.models.univariate.garch import Garch
from components.models.univariate.arch import arch

from components.models.multivariate.mul_prophet import multivariate_prophet_forecast
from components.models.multivariate.mul_var import multivariate_var_forecast
from components.models.multivariate.mult_varmax import multivariate_varmax_forecast


import streamlit as st
import warnings
warnings.filterwarnings("ignore")

st.title("Time Series Analysis")

node, edge = None, None
node = st.sidebar.file_uploader("Upload a Node data file", type=["csv"])
edge = st.sidebar.file_uploader("Upload a Edge data file", type=["csv"])

forecast_type_options = ["Univariate",
                         "Multivariate - Node Level", "Multivariate K-hop"]
selected_forecast_option = st.sidebar.selectbox(
    "Select the model", forecast_type_options)


if node and edge:
    node_data = pd.read_csv(node)
    edge_data = pd.read_csv(edge)

    st.write("Files uploaded successfully")

    if selected_forecast_option == "Univariate":

        options = ["Select", "ARIMA", "STL",
                    # "ARCH", "GARCH",
                   "Prophet", "Exponential Smoothing"]
        selected_option = st.sidebar.selectbox("Select the model", options)

        num_nodes = len(node_data["node"].unique())

        if selected_option == "ARIMA":
            num_features = len(node_data["feature"].unique())
            node_index = st.slider("Select the node index", 0, num_nodes - 1)

            temp = node_data
            st.write("merged data for node", node_index)
            selected_node_data = get_node_data_from_merged(
                merged_data=temp, node_index=node_index)
            st.write(selected_node_data)

            feature_index = st.slider(
                "Select the feature index", 0, num_features - 1)
            selected_node_feature_data = get_feature_with_dates(
                df=selected_node_data, feature_index=feature_index)

            plot = arima_model(data=selected_node_feature_data)
            st.plotly_chart(plot)

        if selected_option == "STL":
            num_nodes = len(node_data["node"].unique())
            num_features = len(node_data["feature"].unique())
            node_index = st.slider("Select the node index", 0, num_nodes - 1)

            temp = node_data
            st.write("merged data for node", node_index)
            # selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
            # st.write(selected_node_data)

            merged_data = node_data
            num_nodes = len(node_data["node"].unique())

            selected_node_data = merged_data[merged_data["node"] == node_index]
            selected_node_data.drop(["node"], axis=1, inplace=True)

            selected_node_data = selected_node_data.pivot(
                index='timestamp', columns='feature', values='value')

            feature_index = st.slider(
                "Select the feature index", 0, num_features - 1)
            selected_node_feature_data = get_feature_with_dates(
                df=selected_node_data, feature_index=feature_index)

            f1 = selected_node_feature_data
            stl = STL(f1, period=5)
            res = stl.fit()
            fig = res.plot()
            

            train_size = int(len(f1) * 0.8)
            test_size = len(f1) - train_size

            stlf = STLForecast(f1[:train_size], ARIMA, model_kwargs=dict(
                order=(3, 1, 3), trend="t"), period=14)
            stlf_res = stlf.fit()
            forecast = stlf_res.forecast(test_size)

            st.pyplot(fig)

            plt.figure()
            plt.plot(f1[train_size:])

            plt.plot(forecast)
            st.pyplot(plt)
            plt.show()

            # final = pd.concat([f1, forecast], axis=1)

            # plot1 = px.line(final, x=final.index, y=final.columns, title="Forecast using STL model")

            model = sm.tsa.arima.ARIMA(f1[:train_size], order=(3, 1, 3))
            model_fit = model.fit()

            forecast = model_fit.forecast(test_size)

            plt.figure()
            plt.plot(f1[train_size:])
            plt.plot(forecast)
            st.pyplot(plt)
            plt.show()

            # st.write(selected_node_feature_data)

            # plot1,plot2,fig = stl_model(f1=selected_node_feature_data)
            # st.plotly_chart(plot1)
            # st.plotly_chart(plot2)

            # forecast using arima model
            # get the acf , pcf plots
            # get the adf test and kpss test
            # based on the above plots set the value of p,d,q
            # run the model using the above values
            # plot the forecast
            # get the error
            # plot the data
            # return the error

        if selected_option == "Prophet":
            results = {}
            rmse_values = []
            merged_data = node_data

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = prophet_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            avg_rmse = np.mean(rmse_values)
            st.metric(f'Average RMSE', avg_rmse)

            node_index = st.slider(
                label="Node ID", min_value=0, max_value=max(num_nodes-1, 1))
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

        if selected_option == "Exponential Smoothing":

            results = {}
            rmse_values = []
            merged_data = node_data

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = ets_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse ': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            rmse_mape = np.mean(rmse_values)
            st.metric(f'Average RMSE', rmse_mape)

            node_index = st.slider(
                label="Node ID", min_value=0, max_value=max(num_nodes-1, 1), key="ets")
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Training, Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

        # if selected_option == "ARCH":
        #     num_features = len(node_data["feature"].unique())
        #     node_index = st.slider("Select the node index", 0, num_nodes - 1)

        #     temp = node_data
        #     st.write("merged data for node", node_index)
        #     selected_node_data = get_node_data_from_merged(
        #         merged_data=temp, node_index=node_index)
        #     st.write(selected_node_data)

        #     plot = arch(selected_node_data)

        # if selected_option == "GARCH":
        #     num_features = len(node_data["feature"].unique())
        #     node_index = st.slider("Select the node index", 0, num_nodes - 1)

        #     temp = node_data
        #     st.write("merged data for node", node_index)
        #     selected_node_data = get_node_data_from_merged(
        #         merged_data=temp, node_index=node_index)
        #     st.write(selected_node_data)

        #     plot = Garch(selected_node_data)

    if selected_forecast_option == "Multivariate - Node Level":

        num_nodes = len(node_data["node"].unique())
        forecast_options = ["Select", "Multivariate Prophet",
                            "Multivariate VAR", "Multivariate VARMAX"]
        selected_option = st.sidebar.selectbox(
            "Select the model", forecast_options)
        merged_data = node_data

        if selected_option == "Multivariate Prophet":
            results = {}
            rmse_values = []

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = multivariate_prophet_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            avg_rmse = np.mean(rmse_values)
            st.metric(f'Average RMSE', avg_rmse)

            node_index = st.slider(
                label="Node ID", min_value=0, max_value=max(num_nodes-1, 1))
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Training, Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

        if selected_option == "Multivariate VAR":
            results = {}
            rmse_values = []

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = multivariate_var_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            avg_rmse = np.mean(rmse_values)
            st.metric(f'Average RMSE', avg_rmse)

            node_index = st.slider(label="Node ID", min_value=0, max_value=max(
                num_nodes-1, 1), key='mul_var')
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Convert the numpy array to a DataFrame
            forecasts_df = pd.DataFrame(forecasts, columns=train.columns)

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Training, Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

        if selected_option == "Multivariate VARMAX":
            results = {}
            rmse_values = []

            for node_index_forecast in range(num_nodes):
                node_data_forecast = get_node_data_from_merged(
                    merged_data=merged_data, node_index=node_index_forecast)
                rmse, train, test, forecasts = multivariate_varmax_forecast(
                    df=node_data_forecast)
                results[node_index_forecast] = {
                    'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                rmse_values.append(rmse)

            avg_rmse = np.mean(rmse_values)
            st.metric(f'Average RMSE', avg_rmse)

            node_index = st.slider(label="Node ID", min_value=0, max_value=max(
                num_nodes-1, 1), key='mul_varmax')
            node_data = get_node_data_from_merged(
                merged_data=merged_data, node_index=node_index)

            st.subheader("Node Data")
            st.dataframe(node_data)

            result = results[node_index]
            train = result['train']
            test = result['test']
            forecasts = result['forecasts']

            # Define a list of colors for the plots
            # Add more colors if needed
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # Convert the numpy array to a DataFrame
            forecasts_df = pd.DataFrame(forecasts, columns=train.columns)

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train)), train[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Training, Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 6))
            for i, column in enumerate(test.columns):
                if column != 'ds':
                    plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(
                        colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                    plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(
                        colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

            plt.title('Actual vs Predicted')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(plt)  # Display the plot in Streamlit

    if selected_forecast_option == "Multivariate K-hop":
        pass
