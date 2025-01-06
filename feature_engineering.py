import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh import select_features


data = pd.read_csv("./input/AEP_hourly.csv")

data["id"] = data.index
train = data[data['Datetime'] < '2016-01-01']
test = data[data['Datetime'] >= '2016-01-01']

train_features_ext = extract_features(train, column_id = "id", column_sort = 'Datetime', column_value = "AEP_MW" ,
                                  n_jobs = 10, default_fc_parameters = EfficientFCParameters())

train_features_ext = train_features_ext.dropna(axis=1, how='all')
train_features_ext = select_features(train_features_ext, train["AEP_MW"])