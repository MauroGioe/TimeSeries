import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.layers import Dense,Dropout,LSTM
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


tf.keras.utils.set_random_seed(123)
train = pd.read_csv("./input/AEP_hourly_train.csv")
test = pd.read_csv("./input/AEP_hourly_test.csv")
target= "AEP_MW"



def normalize_data(df, scaler = None, test=True):
    if test != True:
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df[target].values.reshape(-1,1))
    else:
        normalized_data = scaler.transform(df[target].values.reshape(-1,1))
    df[target] = normalized_data
    return df, scaler

def load_data(data, seq_len, multi_steps):
    X = []
    y = []
    if multi_steps == 1:
        multi_steps_len = multi_steps-1
    else:
        multi_steps_len = multi_steps
    for i in range(seq_len, len(data)-multi_steps_len):
        X.append(data.iloc[i-seq_len : i])
        y.append(data.iloc[i:i+multi_steps])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], seq_len, 1))
    y = np.array(y)
    y = np.reshape(y, (y.shape[0], multi_steps))
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




train_norm, scaler = normalize_data(train, test = False)
test_norm, _ = normalize_data(test, scaler)
train_norm = train_norm[target]
test_norm = test_norm[target]

#set it to 1 to have only 1 prediction per sequence
multi_steps = 1
seq_len = 24
X_train, y_train = load_data(train_norm, seq_len = 24, multi_steps = multi_steps)
X_test, y_test = load_data(test_norm, seq_len = 24, multi_steps = multi_steps)
X_train, X_valid, y_train, y_valid = get_validation_data (X_train, y_train, 0.80)






lstm_model = Sequential()
#input shape = (num_timesteps, num_features)
lstm_model.add(LSTM(48,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(48,activation="tanh",return_sequences=True))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(48,activation="tanh",return_sequences=False))
lstm_model.add(Dropout(0.15))

lstm_model.add(Dense(multi_steps))

lstm_model.summary()



learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss',
                                            patience = 2,
                                            verbose = 1,
                                            factor = 0.5,
                                            min_lr = 0.0001)
earlystop = EarlyStopping(patience = 3, restore_best_weights = True, monitor = "val_loss",  mode = 'min', verbose = 1)

callbacks = [earlystop, learning_rate_reduction]



lstm_model.compile(optimizer = "adam", loss = "MSE")
history = lstm_model.fit(X_train, y_train, epochs = 25, batch_size = 240, callbacks = callbacks,
               validation_data = (X_valid, y_valid))


y_pred = lstm_model.predict(X_test)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_original_scale = scaler.inverse_transform(y_pred).flatten()

rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original_scale))
print (f"The root mean square error is equal to {np.round(rmse, 2)}")
#rmse= 249
mape = mean_absolute_percentage_error(y_test_original, y_pred_original_scale)
#mape = 1.31

r2 = r2_score(y_test, y_pred)
print (f"The R^2 is equal to {np.round(r2, 2)}")
#R^2=0.99



plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show(block = True)

#this last part would have to be modified for multistep prediction

#y values start from sequence length, in this case from 24
results_LSTM = pd.DataFrame({"Date":test["Datetime"][seq_len:], 'Actual': y_test_original,
                             'Predicted': y_pred_original_scale})


#Prediction vs actual values
range_x=[x for x in range(200)]

plt.figure(figsize=(20,6))

plt.plot(range_x, y_test_original[:200], marker='.', label="actual", color='purple')
plt.plot(range_x, y_pred_original_scale[:200], '-', label="prediction", color='red')

plt.ylabel('Global_active_power', size=14)
plt.xlabel('Time step', size=14)
plt.legend(fontsize=16)
plt.show(block = True)