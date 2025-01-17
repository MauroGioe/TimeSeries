import pandas as pd


data = pd.read_csv("./input/AEP_hourly.csv")
data["Datetime"] = pd.to_datetime(data['Datetime'])
data["hour"] = data["Datetime"].dt.hour
data["day_of_week"] = data["Datetime"].dt.dayofweek
data["day_of_month"] = data["Datetime"].dt.day
data['day_of_year'] = data['Datetime'].dt.dayofyear
data["month"] = data["Datetime"].dt.month
data['quarter'] = data['Datetime'].dt.quarter
data["year"] = data["Datetime"].dt.year

data.sort_values('Datetime', inplace=True, ascending=True)
train = data[data['Datetime'] < '2016-01-01']
test = data[data['Datetime'] >= '2016-01-01']



train.to_csv("./input/AEP_hourly_train.csv", index=False)
test.to_csv("./input/AEP_hourly_test.csv", index=False)