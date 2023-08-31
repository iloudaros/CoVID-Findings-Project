import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import datetime as dt


#Data loading and filtering
data = pd.read_csv('data_proccesed.csv')
data = data[data['Entity'] == 'Greece']

#Date modification
start_date = pd.to_datetime('2021-01-01')
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(dt.datetime.toordinal)
#filtered_data = data[data['Date'] >= start_date]

#Select relevant features 
features = ['Daily tests', 'Cases', 'Deaths', 'GDP/Capita', 'Population', 'Median age', 'Hospital beds per 1000 people', 'Medical doctors per 1000 people']
data['Positivity Percentage'] = data['Cases'] / data['Daily tests']

#Calculate positivity percentage 3 days later and modify
data['Target'] = data['Cases'] / data['Daily tests'].shift(-3)
data = data.select_dtypes(include=[np.number])

#Prepare sequences for training and Train-test split
x = data.drop('Target', axis=1)
y = data['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Transform and scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Building the RNN model
RNNmodel = Sequential()
RNNmodel.add(SimpleRNN(units=50, return_sequences=True, input_shape=(1, len(x_train.columns))))
RNNmodel.add(SimpleRNN(units=50, return_sequences=True))
RNNmodel.add(Dense(units=1))

#Compiling the model
RNNmodel.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

#Training the RNN model
x_train_scaled = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
x_test_scaled = x_test_scaled.reshape((x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))
RNNmodel.fit(x_train_scaled, y_train, epochs=150, batch_size=10, verbose=2)

#Making predictions - RNN
RNNpredictions = RNNmodel.predict(x_test_scaled)
print('RNN MSE:', mean_squared_error(y_test, RNNpredictions))
#RNNpredictions = scaler.inverse_transform(RNNpredictions)
#RNNprediction_df = pd.DataFrame(RNNpredictions, columns=['Predicted_Posittivity_Percentage'])
#RNNprediction_df['Actual_Positivity_Percentage'] = y.test.values
#RNNprediction_df.to_csv('gr_predictions.csv', index=False)
#print(head())

#Creating SVM model
SVMmodel = svm.SVR()
SVMmodel.fit(x_train_scaled, y_train)

#Making predictions - SVM
SVMpredictions = SVMmodel.predict(x_test_scaled)
print('SVM MSE:', mean_squared_error(y_test, SVMpredictions))
