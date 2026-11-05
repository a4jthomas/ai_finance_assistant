import streamlit as st 
import pandas as pd

# Import our modules from the src folder
from src.tools import fetch_stock_data, process_data, create_candlestick_chart, create_daily_returns_histogram, create_high_low_chart, create_moving_average_chart, create_volume_vs_close_scatter

# ----------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Enhanced Financial Planning",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Enhanced Financial Planning")
st.caption("This dashboard lets you explore full historical stock data with interactive charts. Use the sidebar to filter data and adjust settings.")

# ----------------------------------------------------
# 2. SIDEBAR CONTROLS, DATA FETCHING & DISPLAY METRICS
# ----------------------------------------------------
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")

if ticker:
    # Fetch the complete historical data
    data = fetch_stock_data(ticker)
    
    if data.empty:
        st.error("No data found for the provided ticker. Double-check the symbol and try again.")
    else:
        # Convert the Date column to datetime and remove timezone info for consistency.
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        
        # Sidebar: Date range filter based on the data's min and max dates
        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()
        start_date, end_date = st.sidebar.date_input(
            "Select Date Range", 
            value=[min_date, max_date],
            min_value=min_date, 
            max_value=max_date
        )
        
        # Filter the data based on the selected date range
        mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
        data = data.loc[mask].copy()
        
        # Sidebar: Moving average window slider
        ma_window = st.sidebar.slider("Moving Average Window", min_value=5, max_value=200, value=50, step=1)
        
        # Process the data to add daily returns and moving average
        data = process_data(data, ma_window)
        
        st.subheader(f"{ticker} Stock Data from {start_date} to {end_date}")
        
        # Display interactive candlestick chart
        st.markdown("### Candlestick Chart")
        fig_candle = create_candlestick_chart(data, ticker)
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Display closing price along with moving average line chart
        st.markdown("### Closing Price with Moving Average")
        fig_ma = create_moving_average_chart(data, ticker, ma_window)
        st.plotly_chart(fig_ma, use_container_width=True)
        
        # Display histogram of daily returns
        st.markdown("### Daily Returns Distribution")
        fig_hist = create_daily_returns_histogram(data)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Display scatter plot of Volume vs. Closing Price
        st.markdown("### Volume vs. Closing Price")
        fig_scatter = create_volume_vs_close_scatter(data)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Display line chart for High and Low prices
        st.markdown("### High and Low Prices Over Time")
        fig_high_low = create_high_low_chart(data)
        st.plotly_chart(fig_high_low, use_container_width=True)
        
        # Show a snapshot of the raw data
        st.markdown("### Data Snapshot")
        st.dataframe(data.head())



