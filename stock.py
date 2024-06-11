import numpy as np
import pandas as pd
import yfinance as yf 
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model("Stock prediction model.keras")

st.header("Stock Market Predictor")

stock = st.text_input("Enter stock Symbol", "GOOG")
start = "2012-01-01"
end = "2022-12-31"

data = yf.download(stock, start, end)

st.subheader("Stock Data")
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

p_100_days = data_train.tail(100)
data_test = pd.concat([p_100_days, data_train], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader("MA50")

ma_50_days = data.Close.rolling(50).mean()
ma_50_chart = pd.DataFrame({'MA50': ma_50_days, 'Close Price': data.Close})
st.line_chart(ma_50_chart)

st.subheader("MA100 and MA250")

ma_100_days = data.Close.rolling(100).mean()
ma_250_days = data.Close.rolling(250).mean()

ma_chart = pd.DataFrame({'MA100': ma_100_days, 'MA250': ma_250_days, 'Close Price': data.Close})
st.line_chart(ma_chart)

x = []
y_true = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y_true.append(data_test_scale[i, 0])
    
x, y_true = np.array(x), np.array(y_true)

predict = model.predict(x)

scale = 1 / scaler.scale_[0]

predict = predict * scale
y_true = y_true * scale

# Plot the predicted and true values using explicit figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_true, label='True Values')
ax.plot(predict, label='Predicted Values')
ax.set_xlabel('Time')
ax.set_ylabel('Scaled Value')
ax.set_title('True vs Predicted Values')
ax.legend()

# Use st.pyplot with the figure object
st.pyplot(fig)
