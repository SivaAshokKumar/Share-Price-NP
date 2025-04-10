import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objs as go
import plotly.express as px
import datetime
from io import BytesIO

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 NeuralProphet Stock Price Predictor")
st.markdown("Predict future stock prices using Facebook's NeuralProphet with multiple stocks comparison.")

# Sidebar Inputs
st.sidebar.header("📊 Prediction Settings")
tickers_input = st.sidebar.text_input("Enter NSE Stock Tickers (comma separated)", value="INFY.NS, TCS.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2023, 10, 1))
forecast_days = st.sidebar.slider("Days to Forecast", min_value=10, max_value=90, value=30)

# CSV Download
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Main Prediction Logic
if st.sidebar.button("🔮 Predict"):
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    results = []

    for ticker in tickers:
        st.subheader(f"📌 {ticker} Prediction Results")

        with st.spinner(f"Fetching data for {ticker}..."):
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.warning(f"No data found for {ticker}.")
                continue

            data = data[['Close']].reset_index()
            data.columns = ['ds', 'y']

            # Train-test split
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

            model = NeuralProphet(
                n_forecasts=1,
                n_lags=60,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )

            model.fit(train_data, freq='D')

            # Evaluation on test set
            future_test = model.make_future_dataframe(test_data, periods=0)
            forecast_test = model.predict(future_test)

            merged = test_data.merge(forecast_test[['ds', 'yhat1']], on='ds', how='inner').dropna()
            y_true = merged['y'].values
            y_pred = merged['yhat1'].values

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            # Predict future
            last_date = data['ds'].max()
            future_data = model.make_future_dataframe(data, periods=forecast_days, n_historic_predictions=False)
            forecast = model.predict(future_data)
            future_forecast = forecast[forecast['ds'] > last_date]

            # Store results
            results.append({
                'ticker': ticker,
                'r2': r2,
                'rmse': rmse,
                'mape': mape,
                'forecast': future_forecast[['ds', 'yhat1']].rename(columns={'ds': 'Date', 'yhat1': 'Predicted Price'})
            })

            # Show forecast table
            st.write("📅 Predicted Future Prices")
            st.dataframe(future_forecast[['ds', 'yhat1']].rename(columns={'ds': 'Date', 'yhat1': 'Predicted Price'}).set_index('Date'))

            # CSV download
            csv = convert_df(future_forecast[['ds', 'yhat1']].rename(columns={'ds': 'Date', 'yhat1': 'Predicted Price'}))
            st.download_button(label="📥 Download Forecast CSV", data=csv, file_name=f"{ticker}_forecast.csv", mime='text/csv')

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Actual', mode='lines'))
            fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['Predicted Price'], name='Forecast', mode='lines+markers'))
            fig.update_layout(title=f'{ticker} - Stock Price Forecast', xaxis_title='Date', yaxis_title='Price (INR)', template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

    # Accuracy comparison plot
    if results:
        st.subheader("📊 Model Evaluation Comparison")
        eval_df = pd.DataFrame({
            'Ticker': [res['ticker'] for res in results],
            'R2': [res['r2'] for res in results],
            'RMSE': [res['rmse'] for res in results],
            'MAPE': [res['mape'] for res in results],
        })

        col1, col2, col3 = st.columns(3)

        with col1:
            fig_r2 = px.bar(eval_df, x='Ticker', y='R2', title="R-squared Score", color='Ticker', text_auto='.2f')
            st.plotly_chart(fig_r2, use_container_width=True)
        with col2:
            fig_rmse = px.bar(eval_df, x='Ticker', y='RMSE', title="Root Mean Squared Error", color='Ticker', text_auto='.2f')
            st.plotly_chart(fig_rmse, use_container_width=True)
        with col3:
            fig_mape = px.bar(eval_df, x='Ticker', y='MAPE', title="Mean Absolute Percentage Error", color='Ticker', text_auto='.2f')
            st.plotly_chart(fig_mape, use_container_width=True)
