#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


# ### Preparing Bitcoin data



df = pd.read_csv('Bitcoin.csv')
df.replace(',','', regex=True, inplace=True)
df.rename(columns={'Date':'ds'}, inplace = True)
df['ds'] = pd.to_datetime(df['ds'])
df.set_index('ds',inplace=True)
df.index.freq = 'D'
df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1,inplace=True)
df.rename(columns={'Close' : 'y'}, inplace = True)
df['y'] = df.y.astype(float)

df_train = df[df.index <= '2022-11-30']
df_val = df[df.index.isin(pd.date_range("2022-12-01", "2022-12-31"))]
df_test = df[df.index.isin(pd.date_range("2023-01-01", "2023-01-31"))]

df.reset_index(inplace=True)
df_train.reset_index(inplace=True)
df_val.reset_index(inplace=True)
df_test.reset_index(inplace=True)


# ### Loading recession data

df_recession = pd.read_csv('recession.csv')
df_recession['DATE'] = pd.to_datetime(df_recession['DATE'])
df_recession.set_index('DATE',inplace=True)
df_recession.index.freq = 'MS'

date_range = pd.date_range(start=df_recession.index[0], end=df_recession.index[-1], freq='D')
daily_recession = pd.DataFrame(index=date_range)
for i, row in df_recession.iterrows():
    month_end = (pd.to_datetime(str(i)) + pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
    month_range = pd.date_range(start=i, end=month_end)
    try:
        daily_value = row['RECPROUSM156N']
        daily_recession.loc[month_range, 'RECPROUSM156N'] = daily_value
    except KeyError:
        month_range = pd.date_range(start=i, end=i)
        daily_recession.loc[month_range, 'RECPROUSM156N'] = daily_value
#        print("Done")




# ### Loading Interest Rate


daily_interest = pd.read_csv('interest.csv')
daily_interest['DATE'] = pd.to_datetime(daily_interest['DATE'])
daily_interest.set_index('DATE',inplace=True)
daily_interest.index.freq = 'D'


df.drop(df.tail(1).index,inplace=True)




daily_recession.drop(daily_recession.tail(1).index,inplace=True)

# ### LSTM Make x_train and y_train for bitcoin



from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import Dense, LSTM, Add, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
import tensorflow as tf


df_train.set_index('ds',inplace=True)
df_val.set_index('ds', inplace=True)
df_test.set_index('ds', inplace=True)
bitcoin_train = df_train.values
bitcoin_val = df_val.values
bitcoin_train_val = np.concatenate((bitcoin_train, bitcoin_val), axis=0)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(bitcoin_train_val)
scaled_train = scaler.transform(bitcoin_train_val)

n_input = 10
generator = TimeseriesGenerator(scaled_train[0:bitcoin_train.shape[0],:], 
                                scaled_train[0:bitcoin_train.shape[0],:], 
                                length=n_input, 
                                batch_size=1)


x_train, y_train = [], []
for i in range(len(generator)):
    x , y = generator[i]
    x_train.append(x)
    y_train.append(y)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2],1))

#print(x_train.shape)
#print(y_train.shape)



val_generator = TimeseriesGenerator(scaled_train[bitcoin_train.shape[0]-n_input:,:], 
                                scaled_train[bitcoin_train.shape[0]-n_input:,:], 
                                length=n_input, 
                                batch_size=1)
x_val, y_val = [], []
for i in range(len(val_generator)):
    x , y = val_generator[i]
    x_val.append(x)
    y_val.append(y)
x_val, y_val = np.array(x_val), np.array(y_val)
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[2],1))


# ### Prepare Exogenous interest rate data


#interest_scaler = MinMaxScaler()
interest_scaler = StandardScaler()
daily_interest_values = daily_interest.values
interest_scaler.fit(daily_interest_values)
scaled_interest = interest_scaler.transform(daily_interest_values)



exo_interest_train =[]
for i in range(n_input, scaled_train[0:bitcoin_train.shape[0],:].shape[0]):
    #exo_interest_train.append(daily_interest.iloc[i-n_input:i, :].values.flatten())
    exo_interest_train.append(scaled_interest[i-n_input:i, :].flatten())


exo_interest_train = np.array(exo_interest_train)

exo_interest_val =[]
for i in range(n_input, scaled_train[bitcoin_train.shape[0]-n_input:,:].shape[0]):
    train_size = scaled_train[:bitcoin_train.shape[0],:].shape[0]
    
    #exo_interest_val.append(daily_interest.iloc[train_size-n_input+i-n_input:train_size-n_input+i, :].values.flatten())
    exo_interest_val.append(scaled_interest[train_size-n_input+i-n_input:train_size-n_input+i, :].flatten())
exo_interest_val = np.array(exo_interest_val)


# ### Prepare Exogenous recession probability data
#recession_scaler = MinMaxScaler()
recession_scaler = StandardScaler()
daily_recession_values = daily_recession.values
recession_scaler.fit(daily_recession_values)
scaled_recession = recession_scaler.transform(daily_recession_values)

exo_recession_train =[]
for i in range(n_input, scaled_train[0:bitcoin_train.shape[0],:].shape[0]):
    #exo_interest_train.append(daily_interest.iloc[i-n_input:i, :].values.flatten())
    exo_recession_train.append(scaled_recession[i-n_input:i, :].flatten())

exo_recession_train = np.array(exo_recession_train)

exo_recession_val =[]
for i in range(n_input, scaled_train[bitcoin_train.shape[0]-n_input:,:].shape[0]):
    train_size = scaled_train[:bitcoin_train.shape[0],:].shape[0]
    
    #exo_interest_val.append(daily_interest.iloc[train_size-n_input+i-n_input:train_size-n_input+i, :].values.flatten())
    exo_recession_val.append(scaled_recession[train_size-n_input+i-n_input:train_size-n_input+i, :].flatten())
    

exo_recession_val = np.array(exo_recession_val)

# ### Prepare Model

from tensorflow import keras

input_shape = (n_input,1)
inputs = Input(shape=input_shape)
y = LSTM(50,activation='gelu', return_sequences=True)(inputs)
y = LSTM(50, activation='gelu')(y)
lstm = y

interest_exog = Input(shape=(exo_interest_train.shape[1],))

y2 = keras.layers.LayerNormalization()(interest_exog)

l1 = Dense(128, activation = 'gelu')(y2) #128
y = concatenate([lstm, l1])
Dense_1 = 64
for i in range(2):
    y = Dense(Dense_1, activation='gelu')(y)   
y = Dense(1, activation='gelu')(y)
interest_block = y




recession_exog = Input(shape=(exo_recession_train.shape[1],))
y2 = keras.layers.LayerNormalization()(recession_exog)
l2 = Dense(16,activation = 'relu')(y2)
y = concatenate([lstm,l2]) 
Dense_2 = 128
for i in range(2):
    y = Dense(Dense_2, activation='gelu')(y)   
y = Dense(1, activation='relu')(y)
recession_block = y

y = concatenate([interest_block,recession_block]) 
y = Dense(1)(y)
model = Model(inputs=[inputs,interest_exog, recession_exog] , outputs=y)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=.00007), loss='mse', metrics='RootMeanSquaredError')


# Downloading the weights from DigitalOcean
import urllib.request
import os

try:
    os.mkdir('./weights')
except  FileExistsError:
    print('Please delete the folder "weights".')
    quit()

checkpoint_url = "https://bitcoinforecasting.nyc3.digitaloceanspaces.com/checkpoints_64_128_best_v2/checkpoint"
checkpointdata_url = "https://bitcoinforecasting.nyc3.digitaloceanspaces.com/checkpoints_64_128_best_v2/my_checkpoint.data-00000-of-00001"
checkpointindex_url = "https://bitcoinforecasting.nyc3.digitaloceanspaces.com/checkpoints_64_128_best_v2/my_checkpoint.index"

directory = "./weights"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.mkdir(directory)

# Download the files
urllib.request.urlretrieve(checkpoint_url, os.path.join(directory, os.path.basename(checkpoint_url)))
urllib.request.urlretrieve(checkpointdata_url, os.path.join(directory, os.path.basename(checkpointdata_url)))
urllib.request.urlretrieve(checkpointindex_url, os.path.join(directory, os.path.basename(checkpointindex_url)))




#loading the weights
model.load_weights('./weights/my_checkpoint')


earlystop_callback = keras.callbacks.EarlyStopping(
    monitor='val_root_mean_squared_error',
    min_delta=0.0,
    patience=10,
    verbose=1,
    mode='min',
    baseline=None,
    restore_best_weights=True
)
epoch = 100
history = model.fit([x_train,exo_interest_train,exo_recession_train],y_train,
                    validation_data=([x_val,exo_interest_val, exo_recession_val],y_val),
                    callbacks=[earlystop_callback],
                    verbose = 0,
                    epochs = epoch)

Best_epich = earlystop_callback.best_epoch


# # Validation Non teacher forcing technique

first_eval_batch = scaled_train[bitcoin_train.shape[0]-n_input:bitcoin_train.shape[0],:]
first_eval_batch = first_eval_batch.reshape(1,first_eval_batch.shape[0],first_eval_batch.shape[1])

test_predictions = []

current_batch = first_eval_batch

for i in range(31):
    exo_interest = exo_interest_val[i].reshape(1,exo_interest_val.shape[1])
    exo_recession = exo_recession_val[i].reshape(1,exo_recession_val.shape[1])
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict([current_batch,exo_interest,exo_recession],verbose=0)[0][0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[[current_pred]]],axis=1)
test_predictions = np.array(test_predictions)

test_predictions = test_predictions.reshape(test_predictions.shape[0],1)
true_predictions = scaler.inverse_transform(test_predictions)

df_true_predictions = pd.DataFrame(true_predictions, 
                                   index=df_val.index[:len(true_predictions)], 
                                   columns=['predicted'])

df_merged = pd.concat([df_val[:len(true_predictions)], df_true_predictions], axis=1)


from statsmodels.tools.eval_measures import rmse
print('RMSE = ', rmse(df_merged['y'], df_merged['predicted']), end='')
print(', Epoch =', Best_epich, end ='')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(', MAPE % = ', mean_absolute_percentage_error(df_merged['y'], df_merged['predicted']))


