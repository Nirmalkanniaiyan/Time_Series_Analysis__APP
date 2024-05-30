import streamlit as st
import pandas as pd
import plotly_express as px
from components.functions import get_node_data_from_merged,get_feature_with_dates
from components.models.arima import arima_model

st.title("Time Series Analysis")

node, edge = None, None
node = st.sidebar.file_uploader("Upload a Node data file", type=["csv"])
edge = st.sidebar.file_uploader("Upload a Edge data file", type=["csv"])

options = ["Select", "ARIMA"]

selected_option = st.sidebar.selectbox("Select the model", options)

if node and edge:
    node_data = pd.read_csv(node)
    edge_data = pd.read_csv(edge)

    st.write("Files uploaded successfully")

    if selected_option == "ARIMA":
        num_nodes = len(node_data["node"].unique())
        num_features = len(node_data["feature"].unique())
        node_index = st.slider("Select the node index", 0, num_nodes - 1)

        temp = node_data
        st.write("merged data for node", node_index)
        selected_node_data = get_node_data_from_merged(merged_data=temp, node_index=node_index)
        st.write(selected_node_data)

        feature_index = st.slider("Select the feature index", 0, num_features - 1)
        selected_node_feature_data = get_feature_with_dates(df=selected_node_data, feature_index=feature_index)        

        plot = arima_model(data=selected_node_feature_data)
        st.plotly_chart(plot)
        
        # forecast using arima model
        # get the acf , pcf plots
        # get the adf test and kpss test
        # based on the above plots set the value of p,d,q
        # run the model using the above values
        # plot the forecast
        # get the error
        # plot the data
        # return the error





