import streamlit as st 
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Real Time Market Data Retriever",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Single Asset - Stock Ticker - Data Retriever")
st.caption("Retrieve the most recent closing price of the provided stock SYMBOL.")

# -----------------------------------------------------
# 2. ADD INPUT, RETRIEVE RESULTS AND BASIC GRAPH
# -----------------------------------------------------

ticker_symbol = st.text_input("Enter a single stock ticker symbol (e.g. AAPL, TSLA, MSFT):")
if ticker_symbol:
    ticker_data = yf.Ticker(ticker_symbol)
    ticker_df = ticker_data.history(period='1mo')

    if not ticker_df.empty:
        st.subheader("Market Data")
        st.subheader(f"Recent Price Action for {ticker_symbol.upper()}")

        #Display line chart of closing prices
        st.line_chart(ticker_df["Close"])

        #Display recent price table
        st.write("### Recent Data Table")
        st.dataframe(ticker_df.tail(10))

    else:
        st.error("No data found for this ticker symbol.  Please try another symbol. Accepts only a single stock symbol.")
    