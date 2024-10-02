import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import yfinance as yf
from plotly import graph_objects as go
import statsmodels.api as sm
import pandas as pd
import numpy as np

with st.sidebar:
    selected = option_menu(
        menu_title="Bitcoin Prediction",
        options=["Home", "Projects", "Simulation"],
        menu_icon="cast",
        default_index=0
    )

def add_css():
    st.markdown(
        """
        <style>
         html, body, [class*="css"]  {
            font-family: 'Montserrat', sans-serif;
        }
        
        .big-font {
            font-size: 45px !important;
            color: #f7931a;
            font-weight: 900;
        }
        
        .med-font {
            font-size: 30px !important;
            font-weight: 900;
        }
        
        .small-font {
            font-size: 16px;
        }
        
        .stSlider label, .stSlider span {
            font-family: 'Montserrat', sans-serif;
        }
        
        .stTextInput label, .stButton button, .stMarkdown p, .stDataframe {
            font-family: 'Montserrat', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_css()

# Function to download Bitcoin and VIX data
def download_data(include_vix=False):
    start_date = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    bitcoin_data = yf.download('BTC-USD', start=start_date, end=end_date)
    bitcoin_data = bitcoin_data[['Adj Close']].rename(columns={'Adj Close': 'Bitcoin_Price'})

    if include_vix:
        vix_data = yf.download('^VIX', start=start_date, end=end_date)
        vix_data = vix_data[['Adj Close']].rename(columns={'Adj Close': 'VIX'})
        data = bitcoin_data.join(vix_data, how='inner')
    else:
        data = bitcoin_data

    data.index = pd.to_datetime(data.index)
    data = data.asfreq('D')
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')

    return data

if selected == "Home":
    st.markdown('<h1 class="big-font">BITCOIN PREDICTION APP</h1>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">An advanced application developed by a group of students to predict Bitcoin prices using machine learning algorithms and sophisticated data analysis. The app aims to assist investors and crypto enthusiasts in making more informed decisions regarding their Bitcoin investments.</p>', unsafe_allow_html=True)

    data = download_data()

    st.markdown(f'<h1 class="med-font">BITCOIN DATA from {data.index.min().strftime("%Y-%m-%d")} to {data.index.max().strftime("%Y-%m-%d")}</h1>', unsafe_allow_html=True)
    st.dataframe(data)

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Bitcoin_Price'], name='Bitcoin Price', line=dict(color='red', width=2)))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    
    plot_raw_data()
    
if selected == "Projects":
    st.markdown('<h1 class="big-font">LETS PREDICT!</h1>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">This feature allows you to leverage advanced prediction models to anticipate market movements and make informed decisions. Additionally, you can customize your view by selecting different types of charts to visualize the predicted data effectively.</p>', unsafe_allow_html=True)

    include_vix = st.checkbox('Include VIX (Volatility Index)')

    data = download_data(include_vix=include_vix)

    n_days = st.slider('Days Of Prediction:', 1, 30)  # Updated max value to 30
    period = n_days

    prices = data['Bitcoin_Price']

    if include_vix:
        exog = data[['VIX']]
    else:
        exog = None

    p, d, q = 1, 1, 1  
    P, D, Q, S = 1, 1, 1, 12  

    model = sm.tsa.statespace.SARIMAX(prices, exog=exog, order=(p, d, q), seasonal_order=(P, D, Q, S))
    model_fit = model.fit(disp=False)

    forecast_steps = period
    
    if include_vix:
        # Pastikan data VIX cukup panjang untuk prediksi yang diminta
        if len(exog) < forecast_steps:
            st.error(f'Data VIX tidak cukup panjang untuk prediksi {forecast_steps} hari. Data VIX hanya memiliki {len(exog)} baris.')
        else:
            forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog[-forecast_steps:])
    else:
        forecast = model_fit.get_forecast(steps=forecast_steps)
    
    forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog[-forecast_steps:] if include_vix else None)
    forecast_index = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    chart_type = st.radio('Select Chart Type:', ('Line Chart', 'Candlestick Chart'))

    if chart_type == 'Line Chart':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Bitcoin_Price'], mode='lines', name='Observed'))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_conf_int.iloc[:, 0], mode='lines', name='Lower Confidence Interval', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_conf_int.iloc[:, 1], mode='lines', name='Upper Confidence Interval', line=dict(color='red', dash='dash'), fill='tonexty'))
        fig.update_layout(
            title='BTC-USD Adjusted Close Price with Forecast',
            xaxis_title='Date',
            yaxis_title='Adjusted Close Price',
            xaxis_rangeslider_visible=True,
            yaxis=dict(
                range=[0, 100000],
                tickvals=[0, 20000, 40000, 60000, 80000, 100000]
            )
        )

    else:
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                            open=data['Open'] if 'Open' in data.columns else data['Bitcoin_Price'],
                                            high=data['High'] if 'High' in data.columns else data['Bitcoin_Price'],
                                            low=data['Low'] if 'Low' in data.columns else data['Bitcoin_Price'],
                                            close=data['Bitcoin_Price'])])
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=forecast_index,
                                y=forecast_conf_int.iloc[:, 0],
                                mode='lines',
                                line=dict(color='red', width=1),
                                name='Lower Confidence Interval',
                                fill='tonexty',
                                fillcolor='rgba(255, 0, 0, 0.3)'))
        fig.add_trace(go.Scatter(x=forecast_index,
                                y=forecast_conf_int.iloc[:, 1],
                                mode='lines',
                                line=dict(color='red', width=1),
                                name='Upper Confidence Interval',
                                fill='tonexty',
                                fillcolor='rgba(255, 0, 0, 0.3)'))
        fig.update_layout(
            title='BTC-USD Candlestick Chart with Forecast',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=True,
            yaxis=dict(
                tickvals=[20000, 40000, 60000, 80000, 100000],
                range=[15000, 105000],
                fixedrange=False
            )
        )

    st.plotly_chart(fig)
if selected == "Simulation":
    st.markdown('<h1 class="big-font">SIMULATION</h1>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Simulate your investment growth over the prediction period based on the forecasted Bitcoin prices. Enter your initial investment amount below to see how it grows over time.</p>', unsafe_allow_html=True)

    include_vix = st.checkbox('Include VIX (Volatility Index)')

    data = download_data(include_vix=include_vix)

    n_days = st.slider('Days Of Prediction:', 1, 30)  # Updated max value to 30
    period = n_days

    prices = data['Bitcoin_Price']

    if include_vix:
        exog = data[['VIX']]
    else:
        exog = None

    p, d, q = 1, 1, 1  
    P, D, Q, S = 1, 1, 1, 12  

    model = sm.tsa.statespace.SARIMAX(prices, exog=exog, order=(p, d, q), seasonal_order=(P, D, Q, S))
    model_fit = model.fit(disp=False)

    forecast_steps = period
    forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog[-forecast_steps:] if include_vix else None)
    forecast_index = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_mean = forecast.predicted_mean

    initial_investment = st.number_input('Initial Investment ($):', min_value=0, value=1000)
    investment_growth = initial_investment * (forecast_mean / forecast_mean[0])

    # Calculate final investment value
    final_investment_value = investment_growth.iloc[-1]

    # Display investment growth chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_index, y=investment_growth, mode='lines', name='Investment Growth', line=dict(color='white')))
    fig.update_layout(
        title='Investment Growth Over Prediction Period',
        xaxis_title='Date',
        yaxis_title='Investment Value ($)',
        xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig)

    # Display final investment value
    st.markdown(f'<h2 class="med-font">Final Investment Value: ${final_investment_value:,.2f}</h2>', unsafe_allow_html=True)
