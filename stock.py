import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load the model
model = load_model("Stock prediction model.keras")

# Streamlit header
st.header("Stock Market Predictor")

# Input for stock symbol
stock = st.text_input("Enter stock Symbol", "GOOG")
start = "2012-01-01"
end = "2022-12-31"

# Download stock data
data = yf.download(stock, start, end)

# Display stock data
st.subheader("Stock Data")
st.write(data)

# Split the data into train and test sets
data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Prepare the data for scaling
p_100_days = data_train.tail(100)
data_test = pd.concat([p_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Display Moving Average
st.subheader("MA50")
ma_100_days = data['Close'].rolling(100).mean()
ma_300_days = data['Close'].rolling(300).mean()
ma_50_days = data['Close'].rolling(50).mean()

fig1 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, "r", label="MA50")
plt.plot(data['Close'], "g", label="Close Price")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{stock} Moving Average 50 and Close Price")
st.pyplot(fig1)

st.subheader("Price vs MA50 vs MA100 vs MA300")
fig3 = plt.figure(figsize=(10, 8))
plt.plot(data['Close'], label="Close Price")
plt.plot(ma_50_days, "r", label="MA50")
plt.plot(ma_100_days, "b", label="MA100")
plt.plot(ma_300_days, "g", label="MA300")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{stock} Price vs Moving Averages")
st.pyplot(fig3)


# Prepare the data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Make predictions
predict = model.predict(x)

# Scale back the predictions
scale = 1 / scaler.scale_[0]
predict = predict * scale
y = y * scale

# Plot the predictions
st.subheader("Actual vs Predicted Prices")
fig2 = plt.figure(figsize=(10, 8))
plt.plot(y, color="blue", label="True Price")
plt.plot(predict, color="red", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title(f"{stock} Price Prediction")
plt.legend()
st.pyplot(fig2)