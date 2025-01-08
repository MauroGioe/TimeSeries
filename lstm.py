import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

train = pd.read_csv("./input/AEP_hourly_train.csv")
test = pd.read_csv("./input/AEP_hourly_test.csv")
target= "AEP_MW"


def normalize_data(df):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[target].values.reshape(-1,1))
    df[target] = normalized_data
    return df

train_norm = normalize_data(train)
test_norm = normalize_data(test)
train_norm = train_norm[target]
test_norm = test_norm[target]
def load_data(data, seq_len):
    X_train = []
    y_train = []
    for i in range(seq_len, len(data)):
        X_train.append(data.iloc[i-seq_len : i])
        y_train.append(data.iloc[i])

    return X_train, y_train

X_train, y_train = load_data(train_norm, 24)
