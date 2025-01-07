import pandas as pd
from xgboost import XGBRegressor


train = pd.read_csv("./input/AEP_hourly_train.csv")
test = pd.read_csv("./input/AEP_hourly_test.csv")


bst = XGBRegressor(n_estimators=2, max_depth=2, learning_rate=1, objective='reg:squarederror')