import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh import select_features


data = pd.read_csv("./input/AEP_hourly.csv")

data["id"] = data.index
train = data[data['Datetime'] < '2016-01-01']
test = data[data['Datetime'] >= '2016-01-01']

train_features_ext = extract_features(train, column_id = "id", column_sort = 'Datetime', column_value = "AEP_MW" ,
                                  n_jobs = 10, default_fc_parameters = MinimalFCParameters())

train_features_ext = train_features_ext.dropna(axis=1, how='all')
train_features_ext = select_features(train_features_ext, train["AEP_MW"])

test_features_ext = extract_features(test, column_id = "id", column_sort = 'Datetime', column_value = "AEP_MW" ,
                                  n_jobs = 10, default_fc_parameters = MinimalFCParameters())

test_features_ext = test_features_ext.dropna(axis=1, how='all')
test_features_ext = select_features(test_features_ext, test["AEP_MW"])

train_features_ext.to_csv("./input/AEP_hourly_train.csv")
test_features_ext.to_csv("./input/AEP_hourly_test.csv")