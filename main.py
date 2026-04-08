import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
from fastapi import FastAPI
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Stock Price Prediction API",
    description="API for predicting stock prices",
    version="1.0"
)


# Load model
import os
print("Files in current dir:", os.listdir())
model = load_model("Stock_model.h5")

@app.get("/")
def home():
    return {"message": "Stock Prediction API Running 🚀"}

@app.get("/predict/{stock}")
def predict_stock(stock: str = "POWERGRID.NS"):
    try:
        start = dt.datetime(2005, 1, 1)
        end = dt.datetime.now()

        df = yf.download(stock, start=start, end=end)

        if df.empty:
            return {"error": "Invalid stock symbol"}

        # EMA
        ema20 = df.Close.ewm(span=20).mean()
        ema50 = df.Close.ewm(span=50).mean()
        ema100 = df.Close.ewm(span=100).mean()
        ema200 = df.Close.ewm(span=200).mean()

        # Train-test split
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80):])

        scaler = MinMaxScaler()
        data_training_array = scaler.fit_transform(data_training)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

        input_data = scaler.transform(final_df)  

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test = np.array(x_test)

        # Prediction
        y_predicted = model.predict(x_test)

        y_predicted = scaler.inverse_transform(y_predicted)


        last_price = float(df.Close.iloc[-1])
        predicted_price = float(y_predicted[-1][0])

        trend = "UP 📈" if predicted_price > last_price else "DOWN 📉"


        # Prepare graph data
        graph_data = {
    "dates": df.index.strftime('%Y-%m-%d').tolist(),
    "close": df["Close"].values.tolist(),
    "ema20": ema20.values.tolist(),
    "ema50": ema50.values.tolist(),
    "ema100": ema100.values.tolist(),
    "ema200": ema200.values.tolist()
}


        return {
    "stock": stock,
    "last_price": last_price,
    "predicted_price": predicted_price,
    "trend": trend,
    "graph": graph_data,
    "status": "success"
}


    except Exception as e:
        return {"error": str(e)}