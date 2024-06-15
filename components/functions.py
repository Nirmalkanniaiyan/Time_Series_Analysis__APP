from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def get_node_data_from_merged(merged_data, node_index):
    filtered = merged_data[merged_data["node"] == node_index]
    filtered.drop(["node"], axis=1, inplace=True)
    unflattened = unflatten_dataframe(filtered)
    return unflattened

def unflatten_dataframe(df_flat):
    df = df_flat.pivot(index='timestamp', columns='feature', values='value')
    start_date = datetime.today().date() - timedelta(days=len(df))
    df["timestamp"] = pd.date_range(start=start_date, periods=len(df), freq='D')
    df.set_index('timestamp', inplace=True)
    return df

def get_feature_with_dates(df,feature_index) :    
    filtered = df.iloc[:,feature_index]
    return filtered

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))