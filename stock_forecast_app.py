import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objs as go
import datetime

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 NeuralProphet Stock Price Predictor")
st.markdown("Predict future stock prices using Facebook's NeuralProphet")

# Sidebar
st.sidebar.header("📊 Prediction Settings")
ticker = st.sidebar.text_input("Enter NSE Stock Ticker", value="INFY.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2023, 10, 1))
forecast_days = st.sidebar.slider("Days to Forecast", min_value=10, max_value=90, value=30)

if st.sidebar.button("🔮 Predict"):
    with st.spinner("Fetching data and training the model..."):
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("No data found for this ticker.")
        else:
            data = data[['Close']].reset_index()
            data.columns = ['ds', 'y']

            # Train-test split
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

            model = NeuralProphet(
                n_forecasts=forecast_days,
                n_lags=60,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )

            metrics = model.fit(train_data, freq='D')

            future = model.make_future_dataframe(test_data, periods=len(test_data))
            forecast = model.predict(future)

            merged = test_data.merge(forecast[['ds', 'yhat1']], on='ds', how='inner')
            merged.dropna(inplace=True)

            y_true = merged['y'].values
            y_pred = merged['yhat1'].values

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            st.subheader("📌 Model Evaluation")
            st.write(f"**R-squared:** {r2:.4f}")
            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**MAPE:** {mape:.2f}%")

            # Forecast future values
            future_full = model.make_future_dataframe(data, periods=forecast_days)
            forecast_full = model.predict(future_full)

            # Plot actual vs forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Actual'))
            fig.add_trace(go.Scatter(x=forecast_full['ds'], y=forecast_full['yhat1'], name='Forecast'))
            fig.update_layout(title='📈 Stock Price Forecast', xaxis_title='Date', yaxis_title='Price (INR)', template='plotly_dark')

            st.subheader("📉 Forecast Visualization")
            st.plotly_chart(fig, use_container_width=True)
