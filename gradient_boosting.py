import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error

train = pd.read_csv("./input/AEP_hourly_train.csv")
test = pd.read_csv("./input/AEP_hourly_test.csv")
target= "AEP_MW"


param_grid = {
    'max_depth': [2, 4, 6],
    'learning_rate': [0.01, 0.5, 1],
    'n_estimators': [10, 100, 200]
}
model =  XGBRegressor(objective = 'reg:squarederror', seed = 123)
tss = TimeSeriesSplit(n_splits = 5, test_size = 24*365)
grid_search = GridSearchCV(estimator = model, cv = tss, param_grid = param_grid)

y_train = train[target]
x_train = train.drop([target, "Datetime"], axis=1)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_


y_test = test[target]
x_test = test.drop([target, "Datetime"], axis=1)

y_pred = best_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

rmse
#rmse= 1774