import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers import LSTM
import dataXY
from keras.utils import plot_model
# FOR REPRODUCIBILITY
np.random.seed(64)
dataset = pd.read_csv('SOCB.csv', usecols=[0])

#dataset = pd.read_csv('vlt5.csv', usecols=[0])
obs = np.arange(1, len(dataset) + 1, 1)
OHLC_avg=dataset
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)

# TRAIN-TEST SPLIT
train_OHLC = int(len(OHLC_avg) * 0.85)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
testX, testY = preprocessing.new_dataset(test_OHLC, 1)

# RESHAPING TRAIN AND TEST DATA
step_size = 3
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# LSTM MODEL
model = Sequential()
#model.add(LSTM(64, input_shape=(1, step_size)))
model.add(LSTM(64, input_shape=(1, step_size),return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences = True))
model.add(Dropout(0.3))
#model.add(LSTM(64,return_sequences = True))
#model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1))
#model.add(Activation('linear'))
#model.add(Activation('tanh'))


# MODEL COMPILING AND TRAINING
model.compile(loss='mean_squared_error', optimizer='adam') # Try SGD, adam, adagrad and compare!!!
model.summary()
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


#plot_model(model, to_file='model.png',show_shapes=True)




# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)


# TRAINING RMSE
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train RMSE: %.2f' % (trainScore))
trainScore2 = mean_absolute_error(trainY, trainPredict)
print('Train MAE: %.2f' % (trainScore2))
# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test RMSE: %.2f' % (testScore))
testScore2=mean_absolute_error(testY, testPredict)
print('Test MAE: %.2f' % (testScore2))


'''
# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict

# DE-NORMALIZING MAIN DATASET
OHLC_avg = scaler.inverse_transform(OHLC_avg)
'''

# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
y=np.concatenate([trainY,testY])
predict=np.concatenate([trainPredict,testPredict])

fig, ax = plt.subplots(1)
ax.plot(y, label='original',color='r')
ax.plot(predict, label = 'prediction',color='b')
#ax.plot(testPredict-test_label, label = 'error')
ax.axvline(x=len(trainPredict)+1,color='k', linestyle='--')
ax.legend(loc='best',fontsize = 14)
ax.set_xlabel('Time sampling',fontsize = 16)
ax.set_ylabel('SOC(%)',fontsize = 16)
plt.show()

fig2=plt.figure(2)
plt.plot((testPredict-testY))
plt.ylabel('Test error',fontsize = 16)
plt.xlabel('Time sampling',fontsize = 16)
#plt.legend(loc='upper right',fontsize = 14)
plt.show()


'''

predict=np.concatenate([trainPredictPlot, testPredictPlot])
fig, ax = plt.subplots(1)
ax.plot(OHLC_avg, label='original',color='r')
ax.plot(predict, label = 'prediction',color='b')
#ax.plot(testPredict-test_label, label = 'error')
ax.axvline(x=len(trainPredictPlot)+1,color='k', linestyle='--')
ax.legend(loc='best')
ax.set_xlabel('time sampling',fontsize = 16)
ax.set_ylabel('SOC(%)',fontsize = 16)
plt.show()

# PREDICT FUTURE VALUES
last_val = testPredict[-1]
last_val_scaled = last_val/last_val
next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
print ("Last Day Value:", np.asscalar(last_val))
print ("Next Day Value:", np.asscalar(last_val*next_val))
# print np.append(last_val, next_val)

'''









