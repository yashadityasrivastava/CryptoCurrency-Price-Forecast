import datetime

import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
import pandas as pd
from keras.models import load_model, Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
import yfinance as yf
import pandas_datareader as web
from statsmodels.tsa.arima_model import ARIMA
from datetime import date
import streamlit as st


end = '2022-05-21'
start = '2018-01-01'

st.title('CryptoCurrency Price Forecast ')

user_input = st.text_input('Enter the Ticker of the Crypto','BTC-USD')

start_date = st.date_input(
      "Enter start date",
     datetime.date(2014, 1, 1))
today = date.today()
dff = web.DataReader(user_input,'yahoo',today)
st.write('Today Date',today)

df = web.DataReader(user_input,'yahoo',start_date,today)


st.write(df.head())

st.write(df.tail())

# Visualisation

st.subheader('Closing Price Vs Time chart')
fig = plt.figure(figsize= (10,7))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Price')
plt.plot(df.Close,'blue')
st.pyplot(fig)


#Spliting the data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.90)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.90): int(len(df))])

to_row2 = int(len(df)*.90)

# split data into train and test
st.subheader('Training and Testing Data')

fig4 = plt.figure(figsize=(10,7))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Price')
plt.plot(df[0:to_row2]['Close'],'green',label = 'train-data')
plt.plot(df[to_row2:]['Close'],'blue',label = 'test-data')
plt.legend()
st.pyplot(fig4)


scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Load Model this is keras-model.h5
#model = Sequential()
#model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
              #input_shape = (x_train.shape[1],1)))
#model.add(Dropout(0.2))

#model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
#model.add(Dropout(0.3))

#model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
#model.add(Dropout(0.4))

#model.add(LSTM(units = 120, activation = 'relu'))
#model.add(Dropout(0.5))

#model.add(Dense(units = 1))

model = load_model('keras-model.h5')

#testing part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index= True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range (100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test) , np.array(y_test)

#Making Prediction
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#st.write("Final Graph")

st.subheader('Prediction vs Original')

st.write('LSTM model')
plt.grid(True)
to_row = int(len(df)*.90)
data_range = df[to_row:].index
plt.title('Price Prediction '+user_input+'LSTM model')
fig2 = plt.figure(figsize=(10,7))
plt.plot(data_range,y_test, 'r' , label = 'Original Price')
plt.plot(data_range,y_predicted, 'b' , label = 'Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.write('ARIMA model')

# train test split
to_row = int(len(df)*.90)

training_data = list(df[0:to_row]['Close'])
testing_data = list(df[to_row:]['Close'])


model_prediction = []
n_test_obs = len(testing_data)

for i in range(n_test_obs):
    model = ARIMA(training_data, order=(4, 2, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = list(output[0])[0]
    model_prediction.append(yhat)
    actual_test_value = testing_data[i]
    training_data.append(actual_test_value)





fig5 =plt.figure(figsize=(15,9))
plt.grid(True)

data_range = df[to_row:].index

plt.plot(data_range, model_prediction,color = 'blue',label = 'predicted price')
plt.plot(data_range, testing_data,color = 'red', label = 'original price')



plt.title('Price Prediction '+user_input +' ARIMA model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.write(fig5)



st.write('Today price of ',user_input,':',dff['Close'][0])
st.write('Future price of ',user_input,':',list(output[0])[0])
















