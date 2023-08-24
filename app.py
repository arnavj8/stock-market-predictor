import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yfin
yfin.pdr_override()
from pandas_datareader import data as pdr
import streamlit as st
from keras.models import load_model
from pandas import concat
from sklearn.preprocessing import MinMaxScaler


bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS01HRgS1ah9pYriD4n0J-fQmaaaFvv-7hG5y5QFehuyzUIuKHopxXrnxm3UKGGHSFRqIo&usqp=CAU");
background-size: cover;
}
</style>
"""
st.markdown(bg_img, unsafe_allow_html=True)

start = '2010-01-01'
end = '2023-08-20'

st.title("Stock Trend Prediction")
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = pdr.get_data_yahoo(user_input, start, end)


#Describe
st.subheader('Data From Jan 1 2010 - August 20 2023')
st.write(df.describe())

def closingPricePredictor():

    #Visualization
    st.subheader("Closing Price v/s Time Chart")
    fig = plt.figure(figsize=(12,8))
    plt.plot(df['Close'])
    st.pyplot(fig)

    #100 moving average
    st.subheader("Closing Price v/s Time Chart with 100 Moving Average")
    ma100 = df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(12,8))
    plt.plot(df['Close'])
    plt.plot(ma100, 'r')
    st.pyplot(fig)

    #200 moving average
    st.subheader("Closing Price v/s Time Chart with 100 Moving Average and 200 Moving Average")
    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12,8))
    plt.plot(df['Close'])
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    st.pyplot(fig)


    # splitting Training and Testing data 
    train_data = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)])
    test_data = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

    scaler = MinMaxScaler(feature_range=(0,1))
    train_data_array = scaler.fit_transform(train_data)

    #loading model
    model = load_model('stock_closingPrice_predictor.h5')

    #testing part
    past_100_days = train_data.tail(100)
    final_df = concat([past_100_days,test_data], ignore_index=True)
    input_data = scaler.fit_transform(final_df)


    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)


    #prediction 
    y_predicted = model.predict(x_test)

    scale_factor = 1/scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    #RSI Indicator
    def calculate_rsi(price_data, period=14):
        delta = np.diff(price_data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate((np.full(period-1, np.nan), rsi))

    rsi_period = 100
    rsi_original = calculate_rsi(y_test, rsi_period)
    y_predicted = y_predicted.ravel()
    rsi_predicted = calculate_rsi(y_predicted, rsi_period)

    #RSI GRAPH
    st.subheader("Predicted RSI v/s Actual RSI")
    fig2 = plt.figure(figsize=(12,8))
    plt.plot(rsi_original, 'teal', label='Original RSI')
    plt.plot(rsi_predicted, 'gold', label="Predicted RSI")
    plt.xlabel('Time')
    plt.ylabel('Range')
    plt.legend()
    st.pyplot(fig2)


    #Final graph
    st.subheader("Predictions v/s Actual")
    fig2 = plt.figure(figsize=(12,8))
    plt.plot(y_test, 'black', label='Original Price')
    plt.plot(y_predicted, 'red', label="Predicted Price")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)





def openingPricePredictor():

    #Visualization
    st.subheader("Opening Price v/s Time Chart")
    fig = plt.figure(figsize=(12,8))
    plt.plot(df['Open'])
    st.pyplot(fig)

    #100 moving average
    st.subheader("Opening Price v/s Time Chart with 100 Moving Average")
    ma100 = df['Open'].rolling(100).mean()
    fig = plt.figure(figsize=(12,8))
    plt.plot(df['Open'])
    plt.plot(ma100, 'r')
    st.pyplot(fig)

    #200 moving average
    st.subheader("Opening Price v/s Time Chart with 100 Moving Average and 200 Moving Average")
    ma200 = df['Open'].rolling(200).mean()
    fig = plt.figure(figsize=(12,8))
    plt.plot(df['Open'])
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    st.pyplot(fig)


    # splitting Training and Testing data 
    train_data = pd.DataFrame(df['Open'][0 : int(len(df)*0.70)])
    test_data = pd.DataFrame(df['Open'][int(len(df)*0.70) : int(len(df))])

    scaler = MinMaxScaler(feature_range=(0,1))
    train_data_array = scaler.fit_transform(train_data)

    #loading model
    model = load_model('stock_openingPrice_predictor.h5')

    #testing part
    past_100_days = train_data.tail(100)
    final_df = concat([past_100_days,test_data], ignore_index=True)
    input_data = scaler.fit_transform(final_df)


    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)


    #prediction 
    y_predicted = model.predict(x_test)

    scale_factor = 1/scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    #Final graph
    st.subheader("Predictions v/s Actual")
    fig2 = plt.figure(figsize=(12,8))
    plt.plot(y_test, 'black', label='Original Price')
    plt.plot(y_predicted, 'red', label="Predicted Price")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)


option = st.radio(
    label="Options",
    options=('Closing Price', 'Opening Price'),
)

if option == 'Closing Price':
    closingPricePredictor()
elif option == 'Opening Price':
    openingPricePredictor()





