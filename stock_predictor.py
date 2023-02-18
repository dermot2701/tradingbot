import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


import alpaca_trade_api as tradeapi

# Set API credentials
api_key = AK3YA8CPTYAM6KTWWVKY
api_secret = tqyM9k5A6ddRRCHi5jAMKOU2AUYRoklj5Nuh2Bzk
base_url = 'https://paper-api.alpaca.markets'

# Create API client
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

#Get account information
account = api.get_account()
print(account)

#Place and order
api.submit_order()
    
# Get open orders
open_orders = api.list_orders(status='open')
print(open_orders)

# Cancel an order
api.cancel_order('ORDER-ID')


# Retrieve stock data
symbol = 'AAPL' # Apple stock
timeframe = '1D' # 1 day intervals
start_date = '2022-01-01'
end_date = '2022-02-01'

# Place a market order for 1 share of AAPL at the latest price
api.submit_order(
    symbol=symbol,
    qty=1,
    side='buy',
    type='market',
    time_in_force='gtc'
)

print("Market order placed for 1 share of AAPL at price: $" + str(latest_price))

   

# Split data into training and testing sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_data, test_data = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Scale data to between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Create input and output datasets for training and testing
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Reshape input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create the model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Test the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Plot the results
plt.plot(df['Close'].values, label='Actual Price')
plt.plot([None for i in train_predict] + [x for x in train_predict], label='Predicted Train Price')
plt.plot([None for i in test_predict] + [x for x in test_predict], label='Predicted Test Price')
plt.title('Yahoo Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
