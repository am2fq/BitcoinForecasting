import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import random
warnings.filterwarnings('ignore')

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
scaler = MinMaxScaler()
df_train[['y']] = scaler.fit_transform(df_train[['y']])
#df_val[['y']] = scaler.fit_transform(df_val[['y']])


df.reset_index(inplace=True)
df_train.reset_index(inplace=True)
df_val.reset_index(inplace=True)
df_test.reset_index(inplace=True)


df_recession = pd.read_csv('recession.csv')
df_recession['DATE'] = pd.to_datetime(df_recession['DATE'])
df_recession.set_index('DATE',inplace=True)
df_recession = df_recession.resample('MS').ffill()
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
        print("not defined")
    


daily_interest = pd.read_csv('interest.csv')
daily_interest['DATE'] = pd.to_datetime(daily_interest['DATE'])
daily_interest.set_index('DATE',inplace=True)
df.drop(df.tail(1).index,inplace=True)
daily_recession.drop(daily_recession.tail(1).index,inplace=True)



daily_recession.index.name='DATE'
extra_df = pd.merge(daily_recession, daily_interest, on='DATE')
extra_df.index.name='ds'
extra_df.rename(columns={'RECPROUSM156N': 'recession', 'DFF': 'interest_rate'},inplace = True)
recession_scaler = StandardScaler()
extra_df['recession'] = recession_scaler.fit_transform(extra_df[['recession']])
interest_scaler = StandardScaler()
extra_df['interest_rate'] = interest_scaler.fit_transform(extra_df[['interest_rate']])




seasonality_prior_scale =8.09566634972648 
changepoint_prior_scale =0.0147192207453614 
n_changepoints = 14 
model_prophet = Prophet(n_changepoints=n_changepoints, changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_prior_scale=seasonality_prior_scale)
model_prophet.add_regressor('interest_rate')
model_prophet.add_regressor('recession')
model_prophet.fit(df_train.merge(extra_df[['recession','interest_rate']], on='ds', how='left'))
future = model_prophet.make_future_dataframe(periods=31)
future = pd.merge(future, extra_df, on='ds')
forecast_val = model_prophet.predict(future)
forecast_val[['yhat', 'yhat_lower', 'yhat_upper']] = scaler.inverse_transform(forecast_val[['yhat', 'yhat_lower', 'yhat_upper']])
rmse = np.sqrt(mean_squared_error(y_true=df_val['y'],
           y_pred=forecast_val['yhat'].iloc[-31:]))
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f'RMSE = {rmse},',end = '')
print('MAPE =', mean_absolute_percentage_error(df_val['y'], forecast_val['yhat'].iloc[-31:]))

