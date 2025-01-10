import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
matplotlib.interactive(True)

from sklearn.inspection import permutation_importance


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




train = pd.read_csv("./input/AEP_hourly_train.csv")
test = pd.read_csv("./input/AEP_hourly_test.csv")
target= "AEP_MW"


param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6],
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
print (f"The root mean square error is equal to {np.round(rmse)}")
#rmse= 1796


mape = mean_absolute_percentage_error(y_test, y_pred)
print (f"The mean absolute percentage error is equal to {np.round(mape)}")
#mape = 9%

r2 = r2_score(y_test, y_pred)
print (f"The R^2 is equal to {np.round(r2, 2)}")
#R^2=0.47

perm_importance = permutation_importance(best_model, x_test, y_test, scoring = "neg_root_mean_squared_error")
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(x_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")