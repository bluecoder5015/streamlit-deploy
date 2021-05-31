import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("Datasets/MSFT.csv")
df2 = df
df2["Date"] = pd.to_datetime(df2["Date"],format="%Y-%m-%d")

pickle.dump(df2, open('df_msft.pkl', 'wb'))

# ## MODEL  (GRU)
training_set=df2['Open']
training_set=pd.DataFrame(training_set)
training_set

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = scaler.fit_transform(training_set)
pickle.dump(scaler, open('scalerMSFT.pkl', 'wb'))


training_size = df2.shape[0]*0.80
testing_size = df.shape[0] - training_size

import numpy as np
X_train = []
y_train = []
for i in range(60, 8844):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))


#Testing 

import numpy as np
X_test = []
y_test = []
date=[]
for i in range(60, 8805):
    X_test.append(training_set_scaled[i-60:i, 0])
    y_test.append(training_set_scaled[i, 0])
    date.append(df2.Date[i])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))


#Building the REAL Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import GRU
from tensorflow import keras


#GRU model with 128 units and 4 layers
model = Sequential()
model.add(GRU(units = 128,activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(GRU(units = 128,activation = 'relu', return_sequences = True,dropout=0.2))
model.add(GRU(units = 128,activation = 'relu', return_sequences = True,dropout=0.2))
model.add(GRU(units = 128,activation = 'relu',dropout=0.2))

# Adding the output layer
model.add(Dense(units = 1))


# In[32]:


model.summary()

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics='accuracy')
# Fitting the GRU to the Training set
model.fit(X_train, y_train,validation_data=(X_test,y_test), epochs = 1, batch_size = 64)


# save the model to disk
#pickle.dump(model, open('modelMSFT.pkl', 'wb'))
model.save("modelMSFT.h5")


# ## Future Prediction
##For testing purpose
f_test =[]
f_test.append(training_set_scaled[8797:8857, 0])

with open('ftestMSFT.pkl', 'wb') as f:
    pickle.dump(f_test, f)

