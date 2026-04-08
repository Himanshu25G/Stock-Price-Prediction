import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Predictor Using LSTM", layout="centered")

# Load model
try:
    model = load_model("Stock_model.h5")
except:
    st.error("Model file not found. Please upload Stock_model.h5")
    st.stop()



st.title("📈 Stock Price Prediction App")

stock = st.text_input("Enter Stock Symbol (e.g. POWERGRID.NS)", "POWERGRID.NS")

if st.button("See Prediction"):

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.now()

    df = yf.download(stock, start=start, end=end)

    if df.empty:
        st.error("Invalid Stock Symbol")
    else:
        st.subheader("Stock Sample Data from 2000 to Present")
        st.write(df.sample(7))

        # EMA
        ema20 = df.Close.ewm(span=20).mean()
        ema50 = df.Close.ewm(span=50).mean()
        ema100 = df.Close.ewm(span=100).mean()
        ema200 = df.Close.ewm(span=200).mean()

        # Plot EMA
        fig, ax = plt.subplots()
        ax.plot(df.Close, label="Close")
        ax.plot(ema50, label="EMA 50")
        ax.plot(ema100, label="EMA 100")
        ax.plot(ema200, label="EMA 200")
        ax.set_title(f"{stock} Price with EMA")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid()
        st.pyplot(fig)




        # ML Prediction
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80):])

        scaler = MinMaxScaler(feature_range=(0, 1))

        scaler.fit(data_training)


        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)


        data_training_array = scaler.transform(data_training)
        input_data = scaler.transform(final_df)


        x_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])

        x_test = np.array(x_test)
        # Ensure 3D shape for LSTM input
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


        y_predicted = model.predict(x_test)

        y_predicted = scaler.inverse_transform(y_predicted)




        st.subheader("📊 Prediction vs Actual")

        fig2, ax2 = plt.subplots(figsize=(10,5))

        dates=df.index[int(len(df)*0.80):]
        ax2.plot(dates, data_testing.values, label="Actual", color='blue')
        ax2.plot(dates, y_predicted, label="Predicted", color='red')
        ax2.set_title(f"{stock} Actual vs Predicted Price")
        ax2.legend()
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.grid(True)
        st.pyplot(fig2)


        st.subheader("📈 Latest Predicted Price")
        st.success(f"Latest Predicted Price: {y_predicted[-1][0]:.2f}")

        st.subheader("📉 Latest Actual Price")
        st.success(f"Latest Actual Price: {data_testing.values[-1][0]:.2f}")


        st.subheader("📊 RMSE of Predictions")
        from sklearn.metrics import mean_squared_error

        rmse= np.sqrt(mean_squared_error(data_testing, y_predicted))
        st.success(f"RMSE: {rmse:.2f}")