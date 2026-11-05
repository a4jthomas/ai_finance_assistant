"""
Agent Tools — the concrete actions agents can take.

Financial Q&A Agent (1 tool):
  • get_finance_answers  – RAG semantic search using our private vector database

Market Analysis Agent (1 tool):
  • get_market_data  – API tool call using yFinance to retrieve current stock prices

Tax Education Agent (1 tool):
  • get_tax_info  – RAG semantic search for tax-related information lookup
 
"""

from __future__ import annotations

import random
import time

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from langchain_core.tools import tool


import streamlit as st
from src.config import get_logger, llm, fin_vector_db


logger = get_logger("tools")


# ═══════════════════════════════════════════════════════════
#  FINANCIAL EDUCATION ANSWERS TOOL - Main Query
# ═══════════════════════════════════════════════════════════

@tool
@st.cache_data
def get_finance_answers(query: str) -> str:
    """Search the FinAsst finance database using semantic search (RAG).

    Args:
        query: natural-language search, e.g. "what is the current interest rate for home loans?"
    """
    logger.info("get_finance_answers  query=%r", query)

    try:
        docs = fin_vector_db(query, k=3)
        if not docs:
            return "No finance information found matching your query."
        results = "Found the following finance information:\n\n"
        for i, doc in enumerate(docs, 1):
            results += f"Product {i}:\n{doc.page_content}\n\n"
        return results
    except Exception as exc:
        logger.exception("Finance search failed")
        return f"Error searching finance information: {exc}"


# ═══════════════════════════════════════════════════════════
#  TAX ANSWERS TOOL
# ═══════════════════════════════════════════════════════════
@tool
@st.cache_data
def get_tax_info(query: str) -> str:
    """Search using the open-source ai ChatGPT model gpt-4 for semantic search (RAG).

    Args:
        query: natural-language search, e.g. "what is the current income tax rate for individuals in Arkansas?"
    """
    logger.info("get_tax_info  query=%r", query)

    try:
        docs = llm.search(query, k=3)
        if not docs:
            return "No tax information found matching your query."
        results = "Found the following tax information:\n\n"
        for i, doc in enumerate(docs, 1):
            results += f"Tax Info {i}:\n{doc.page_content}\n\n"
       
        return results
    except Exception as exc:
        logger.exception("Tax search failed")
        return f"Error searching tax information: {exc}"


# ═══════════════════════════════════════════════════════════
#  MARKET DATA TOOL
# ═══════════════════════════════════════════════════════════
@tool
@st.cache_data
def get_market_data(query: list[str]) -> str:
    """Use the yFinance API to retrieve market data for a given stock symbol or list of symbols from the main query.

    Args:
        query: API query is the stock symbol or symbols only, e.g. ["MSFT", "AAPL"]
    """
    logger.info("get_market_data  query=%r", query)
    try:
        # Download 2 days of data for each ticker in tickerStrings, add a 'ticker' column for identification, and concatenate into a single DataFrame with continuous indexing.
        df = pd.concat([yf.download(ticker, group_by="Ticker", period='2d').assign(ticker=ticker) for ticker in query], ignore_index=True)
        st.subheader("Market Data")
        st.table(df)
        results = "Found the following market data:\n\n" + df.to_string()
        return results
    except Exception as exc:
        logger.exception("Market data search failed")
        return f"Error searching market data: {exc}"


# ═══════════════════════════════════════════════════════════
#  STREAMLIT WIDGET  VISUALIZATION TOOLS
# ═══════════════════════════════════════════════════════════   
# ═══════════════════════════════════════════════════════════
#  PORTFOLIO ANALYSIS TOOLS - Widget Tools
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=1800)
def simulate_efficient_frontier(ann_rets, cov_mat, num_sims, rf_rate, asset_names):
    num_assets = len(asset_names)
    results = np.zeros((3 + num_assets, num_sims))
    
    for i in range(num_sims):
        # Generate random normalized weight allocations
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Portfolio Expected Return and Volatility calculations
        p_return = np.sum(weights * ann_rets)
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
        
        # Risk-adjusted Sharpe metric
        p_sharpe = (p_return - rf_rate) / p_volatility
        
        results[0, i] = p_return
        results[1, i] = p_volatility
        results[2, i] = p_sharpe
        
        # Store individual asset weight vectors
        for w_idx in range(num_assets):
            results[3 + w_idx, i] = weights[w_idx]
            
    # Pack metrics cleanly into structured DataFrame columns
    columns = ['Return', 'Volatility', 'Sharpe Ratio'] + [f'Weight_{asset}' for asset in asset_names]
    sim_df = pd.DataFrame(results.T, columns=columns)
    return sim_df


# Cache historical data downloads for the widget tools in the portfolio analysis section on Streamlit app
@st.cache_data(ttl=3600)
def download_portfolio_data(ticker_list, start, end):
    if not ticker_list:
        return pd.DataFrame(), "No tickers specified."
    try:
        # 1. Download data grouped by ticker to avoid multi-index parsing errors
        raw_data = yf.download(ticker_list, start=start, end=end, group_by='ticker')
        
        if raw_data.empty:
            return pd.DataFrame(), "Returned dataset is empty."
            
        # 2. Safely extract closing prices for each ticker dynamically
        closing_prices = {}
        for ticker in ticker_list:
            if ticker in raw_data.columns.levels[0]:
                # Prefer 'Adj Close', fall back to 'Close' if unavailable
                if 'Adj Close' in raw_data[ticker].columns:
                    closing_prices[ticker] = raw_data[ticker]['Adj Close']
                elif 'Close' in raw_data[ticker].columns:
                    closing_prices[ticker] = raw_data[ticker]['Close']
            else:
                return pd.DataFrame(), f"Ticker '{ticker}' not found in downloaded data."
                
        # 3. Compile back into a clean, flat DataFrame
        data = pd.DataFrame(closing_prices)
        return data, None
        
    except Exception as e:
        return pd.DataFrame(), str(e)


# ═══════════════════════════════════════════════════════════
#  STREAMLIT WIDGET PLANNING VISUALIZATION TOOLS
# ═══════════════════════════════════════════════════════════   
#--- Data Fetching Functions for the widget that takes a single ticker symbol ---
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """
    Fetches the complete historical data for the specified ticker from Yahoo Finance.
    
    Parameters:
        ticker (str): The stock ticker symbol.
    
    Returns:
        pd.DataFrame: A dataframe with the stock's historical data.
    """
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(period="max")
    # Reset index so that 'Date' becomes a column
    data.reset_index(inplace=True)
    return data

#----Data Processing Functions ---
def process_data(data: pd.DataFrame, ma_window: int = 50) -> pd.DataFrame:
    """
    Processes the raw stock data by adding daily returns and a moving average.
    
    Parameters:
        data (pd.DataFrame): The raw stock data with a 'Close' column.
        ma_window (int): The window size for the moving average (default is 50 days).
    
    Returns:
        pd.DataFrame: The data with new columns 'Daily Return' and 'MA'.
    """
    # Calculate daily percentage change in the closing price
    data['Daily Return'] = data['Close'].pct_change()
    # Calculate the moving average on the closing price
    data['MA'] = data['Close'].rolling(window=ma_window).mean()
    return data

# ── Widget Planning Visualization Tools for plotting stock data ───────────────────────────────────────────

def create_candlestick_chart(data, ticker):
    """
    Builds an interactive candlestick chart.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=ticker
    )])
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        title=f"{ticker} Candlestick Chart"
    )
    return fig

def create_moving_average_chart(data, ticker, ma_window):
    """
    Creates a line chart showing the closing price and its moving average.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name="Close",
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'].rolling(window=ma_window).mean(),
        mode='lines',
        name=f'{ma_window}-Day Moving Average',
        line=dict(color='orange')
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        title=f"{ticker} Closing Price with {ma_window}-Day Moving Average"
    )
    return fig
def create_daily_returns_histogram(data):
    """
    Generates a histogram for the daily returns.
    """
    fig = go.Figure(data=[go.Histogram(x=data["Daily Return"], nbinsx=50)])
    fig.update_layout(title="Daily Returns Distribution", xaxis_title="Daily Return")
    return fig

def create_volume_vs_close_scatter(data):
    """
    Creates a scatter plot of trading volume versus closing price.
    """
    fig = go.Figure(data=[go.Scatter(
        x=data["Volume"],
        y=data["Close"],
        mode="markers",
        text=data["Date"],
        hovertemplate="Date: %{text}<br>Volume: %{x}<br>Close: %{y}<extra></extra>"
    )])
    fig.update_layout(title="Volume vs. Closing Price",
                      xaxis_title="Volume", yaxis_title="Close")
    return fig

def create_high_low_chart(data):
    """
    Produces a line chart displaying high and low prices over time.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'],
                             mode='lines', name="High"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'],
                             mode='lines', name="Low"))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        title="High and Low Prices Over Time"
    )
    return fig