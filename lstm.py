import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error

train = pd.read_csv("./input/AEP_hourly_train.csv")
test = pd.read_csv("./input/AEP_hourly_test.csv")
target= "AEP_MW"



def normalize_data(df):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[target].values.reshape(-1,1))
    df[target] = normalized_data
    return df

def load_data(data, seq_len):
    X = []
    y = []
    for i in range(seq_len, len(data)):
        X.append(data.iloc[i-seq_len : i])
        y.append(data.iloc[i])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], seq_len, 1))
    y = np.array(y)
    return X, y


def get_validation_data(x_train, y_train, split):
    split_idx = int(x_train.shape[0] * split)
    x_valid = x_train[split_idx:, :]
    x_train = x_train[:split_idx, :]
    y_valid = y_train[split_idx:]
    y_train = y_train[:split_idx]
    return x_train, x_valid, y_train, y_valid




def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




train_norm = normalize_data(train)
test_norm = normalize_data(test)
train_norm = train_norm[target]
test_norm = test_norm[target]

X_train, y_train = load_data(train_norm, 24)
X_test, y_test = load_data(test_norm, 24)
X_train, X_valid, y_train, y_valid = get_validation_data (X_train, y_train, 0.80)






lstm_model = Sequential()

lstm_model.add(LSTM(48,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(48,activation="tanh",return_sequences=True))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(48,activation="tanh",return_sequences=False))
lstm_model.add(Dropout(0.15))

lstm_model.add(Dense(1))

lstm_model.summary()



learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss',
                                            patience = 2,
                                            verbose = 1,
                                            factor = 0.5,
                                            min_lr = 0.0001)
earlystop = EarlyStopping(patience = 3, restore_best_weights = True, monitor = "val_loss",  mode = 'min')

callbacks = [earlystop, learning_rate_reduction]



lstm_model.compile(optimizer = "adam", loss = "MSE")
lstm_model.fit(X_train, y_train, epochs = 20, batch_size = 240, callbacks = callbacks,
               validation_data = (X_valid, y_valid))


y_pred = lstm_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print (f"The root mean square error is equal to {np.round(rmse, 2)}")
#rmse= 0.03
#mape = mean_absolute_percentage_error(y_test, y_pred)
#divide by 0

