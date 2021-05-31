
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import streamlit as st
import pickle


df = pd.read_csv("Datasets/DataFrame.csv")
df2 = df

df2["Unnamed: 7"].isnull().sum()


df2["DateAndTime"] = df2["Date"].astype(str) + df2["Time"]

df2["DateAndTime"] = pd.to_datetime(df2["DateAndTime"],format="%Y%m%d%H:%M")

df2 = df2.drop(["Unnamed: 7","Date","Time"],axis=1)

##MODEL 
pickle.dump(df2, open('df_nifty.pkl', 'wb'))

training_set=df2['open']
training_set=pd.DataFrame(training_set)
training_set


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(training_set)


# In[71]:


training_size = df2.shape[0]*0.80
testing_size = df.shape[0] - training_size


# In[72]:


##Creating Training Set with step of 60-minutes


# In[73]:


import numpy as np
X_train = []
y_train = []
for i in range(60, 18244):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))


#Testing set


import numpy as np
X_test = []
y_test = []
date=[]
#18305
for i in range(18244, 22805):
    X_test.append(training_set_scaled[i-60:i, 0])
    y_test.append(training_set_scaled[i, 0])
    date.append(df2.DateAndTime[i])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))



#Building the REAL Model
from keras.models import Sequential
from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from keras.optimizers import SGD


# In[38]:


#GRU model with 128 units and 4 layers
model = Sequential()
model.add(GRU(units = 50,activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(GRU(units = 50,activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))
#model.add(GRU(units = 128,activation = 'relu', return_sequences = True,dropout=0.2))
model.add(GRU(units = 50,activation = 'relu'))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))


# In[39]:


model.summary()

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mse',)

# Fitting the RNN to the Training set
history = model.fit(X_train, y_train, epochs=5)
# save the model to disk
model.save("model.h5")
pickle.dump(scaler, open('scaler.pkl', 'wb'))

##For testing purpose
f_test =[]
f_test.append(training_set_scaled[22745:22805, 0])

with open('ftest.pkl', 'wb') as f:
    pickle.dump(f_test, f)