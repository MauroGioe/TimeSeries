import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

train = pd.read_csv("./input/AEP_hourly_train.csv")
test = pd.read_csv("./input/AEP_hourly_test.csv")
target= "AEP_MW"

tss = TimeSeriesSplit(n_splits = 5, test_size = 24*365, gap = 24)

tt=tss.split(train)[0]

for train_index, val_index in tss.split(train):
    print(train_index,val_index)
    train =train.loc[train_index]
    val = train.loc[val_index]

    y_train = train[target]
    x_train = train.drop([target,"Datetime"], axis = 1)

    y_val = val[target]
    x_val = val.drop([target,"Datetime"], axis = 1)


bst = XGBRegressor(n_estimators=2, max_depth=2, learning_rate=1, objective='reg:squarederror')