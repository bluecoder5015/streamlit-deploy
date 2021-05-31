import streamlit as st
import numpy as np
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

st.write('''
    # Simple Stock Price Prediction 
    ''')
st.sidebar.header('Users Input Parameters')
df = pd.DataFrame({
  'first column': ["MSFT 1- Day","NIFTY 1-Minute"]
})
option = st.sidebar.selectbox(
    'Select prediction Interval',
     df['first column'])



if option == "NIFTY 1-Minute":
    'You selected:', option
    df = pickle.load(open('df_nifty.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model = load_model('model.h5')
    with open('ftest.pkl', 'rb') as f:
        f_test = pickle.load(f)

    f_test = np.array(f_test)
    f_test = np.reshape(f_test, (f_test.shape[0], f_test.shape[1],1))

    st.sidebar.header('User Input Parameters')

    def user_input_features():
        date = st.sidebar.date_input('Select Date', datetime.date(2021,4,1))
        date = date.day
        hour = st.sidebar.slider('Hour of the day',9,17,9)
        minute = st.sidebar.slider('Minute of the day',0,59,5)

        return date,hour,minute

    day,hour,minute = user_input_features()

    f_predict = []
    n_days = 1
    minutes = 0
    hours = 8 #[9--17]

    for i in range (n_days):
        while (hours > 0):
            while (minutes < 60):
                res = model.predict(f_test)
                f_predict.append(res[0][0])
                f_test = np.delete(f_test,[0],None)
                f_test = np.append(f_test,res[0][0],None)
                f_test = f_test.reshape(1,60,1)
                minutes=minutes+1
            hours=hours-1
            minutes = 0
        hours=8
        minutes=0


    hour=hour-8

    st.header("Prediction:")
    res =scaler.inverse_transform([[f_predict[day*hour*minute]]])
    col1, col2,col3 = st.beta_columns(3)

    original = df.open[0]
    col1.subheader("Open Price")
    col1.write(res[0][0])

    grayscale = df.open[0]
    col2.subheader("Prev Day Open")
    a=scaler.inverse_transform([[f_predict[day-2]]])[0][0]
    col2.write(a)

    return1 = (a - res[0][0])/(a)*100
    #fo = "{:.2f}".format(return1)
    col3.subheader("Return %")
    col3.write(return1)

    st.header("Visualisation:")
    idx = pd.date_range("2021-04-01", periods=len(f_predict), freq="D")
    ts = pd.Series(range(len(idx)), index=idx)
    r = scaler.inverse_transform([f_predict]).reshape(-1,1)
    r = r.reshape(r.shape[0])
    fig, ax = plt.subplots()
    ax=sns.lineplot(x=df.DateAndTime,y=df['open'],color='r')
    ax=sns.lineplot(x=idx,y=r)
    st.pyplot(fig)



# MSFFTT
if option == "MSFT 1- Day":
    'You selected:', option
    df = pickle.load(open('df_msft.pkl','rb'))
    scaler = pickle.load(open('scalerMSFT.pkl', 'rb'))
    model = load_model('modelMSFT.h5') 
    #model = joblib.load('modelMSFT.pkl')

    with open('ftestMSFT.pkl', 'rb') as f:
        f_test = pickle.load(f)

    f_test = np.array(f_test)
    f_test = np.reshape(f_test, (f_test.shape[0], f_test.shape[1],1))
    def user_input_features():
        date = st.sidebar.date_input('Select Date', datetime.date(2021,5,3))
        #st.write(date.day)
        return date.day


    day = user_input_features()
  
    f_predict = []
    n_days = day

    for i in range (n_days):
        res = model.predict(f_test)
        f_predict.append(res[0][0])
        f_test = np.delete(f_test,[0],None)
        f_test = np.append(f_test,res[0][0],None)
        f_test = f_test.reshape(1,60,1)


    st.header("Prediction:")
    res = scaler.inverse_transform([[f_predict[day-1]]])
    col1, col2,col3 = st.beta_columns(3)

    original = df.Open[0]
    col1.subheader("Open Price")
    col1.write(res[0][0])

    grayscale = df.Open[0]
    col2.subheader("Prev Day Open")
    a=scaler.inverse_transform([[f_predict[day-2]]])[0][0]
    col2.write(a)

    return1 = (a - res[0][0])/(a)*100
    #fo = "{:.2f}".format(return1)
    col3.subheader("Return %")
    col3.write(return1)

    st.header("Visualisation:")
    idx = pd.date_range("2021-05-04", periods=len(f_predict), freq="D")
    ts = pd.Series(range(len(idx)), index=idx)
    r = scaler.inverse_transform([f_predict]).reshape(-1,1)
    r = r.reshape(r.shape[0])
    fig, ax = plt.subplots()
    ax=sns.lineplot(x=df.Date[df.Date.dt.year > 2019],y=df['Open'],color='r')
    ax=sns.lineplot(x=idx,y=r)
    st.pyplot(fig)



## Credits

if st.sidebar.button("Credits"):
    st.sidebar.markdown("<h1 style='text-align: left; color: green;'>Welcome!</h1>",
            unsafe_allow_html=True)
    st.sidebar.subheader("Under Guidance of")
    st.sidebar.info(
        """
        1. Yasin Sir\n
        2. Team @ [Technocolab](https://www.linkedin.com/company/technocolabs/)\n
        """)
    st.sidebar.subheader("Contributors/Project Team")
    st.sidebar.info(
        "1. [Nayana](https://www.linkedin.com/in/)\n"
        "2. [Harshit Singh](https://www.linkedin.com/in/harshit-singh-097/)\n"
        "3. [Yogendra](https://www.linkedin.com/in/)\n"
        "4. [Snehashish](https://www.linkedin.com/in/)\n"
        "5. [Pranay](https://www.linkedin.com/in//)"
    )
    st.sidebar.info("[contact us](https://www.technocolabs.tech/)\n")