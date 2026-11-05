import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.tools import simulate_efficient_frontier, download_portfolio_data
# ----------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Multi-Asset Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Multi-Asset Portfolio Allocation & Optimization Workspace")
st.caption("Analyze historical asset correlations and calculate optimal risk-adjusted weights using Modern Portfolio Theory.")

# ----------------------------------------------------
# 2. SIDEBAR CONTROLS & DATA FETCHING
# ----------------------------------------------------
with st.sidebar:
    st.header("Workspace Parameters")
    
    # Input multiple tickers separated by commas
    ticker_string = st.text_input("Enter Ticker Symbols (comma-separated)", value="WMT, AAPL, MSFT, GOOG")
    tickers = [t.strip().upper() for t in ticker_string.split(",") if t.strip()]
    
    # Date windowing
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * 3) # Default to 3 years for asset optimization
    
    date_range = st.date_input(
        "Historical Optimization Window",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    st.markdown("---")
    st.header("Optimization Settings")
    num_portfolios = st.number_input("Number of Simulation Runs", min_value=500, max_value=20000, value=5000, step=500)
    risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.2, step=0.1) / 100

# Safe tuple unpacking for dates
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = date_range
else:
    start_dt, end_dt = start_date, end_date


if len(tickers) < 2:
    st.warning("Please enter at least two valid ticker symbols to perform portfolio allocation optimizations.")
    st.stop()

adj_close_df, error = download_portfolio_data(tickers, start_dt, end_dt)

if error or adj_close_df.empty:
    st.error(f"Error downloading asset historical metrics. Details: {error}")
    st.stop()

# Drop rows with global missing data to establish clean covariance matrix
adj_close_df = adj_close_df.dropna()

# ----------------------------------------------------
# 3. FINANCIAL MATRICES CALCULATION
# ----------------------------------------------------
# Compute daily returns
daily_returns = adj_close_df.pct_change().dropna()

# Annualized mean returns and covariance matrix (assuming 252 trading days)
annual_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252

# ----------------------------------------------------
# 4. MONTE CARLO PORTFOLIO SIMULATION
# ----------------------------------------------------

simulated_portfolios = simulate_efficient_frontier(annual_returns, cov_matrix, num_portfolios, risk_free_rate, tickers)

# Locate critical optimal optimization coordinates
max_sharpe_portfolio = simulated_portfolios.iloc[simulated_portfolios['Sharpe Ratio'].idxmax()]
min_vol_portfolio = simulated_portfolios.iloc[simulated_portfolios['Volatility'].idxmin()]

# ----------------------------------------------------
# 5. DASHBOARD LAYOUT & VISUALIZATIONS
# ----------------------------------------------------
st.subheader("Optimal Allocation Configurations")

# Interactive Layout Columns
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.write("### 🎯 Maximum Sharpe Ratio Portfolio")
    st.metric("Expected Return", f"{max_sharpe_portfolio['Return']*100:.2f}%")
    st.metric("Annual Volatility", f"{max_sharpe_portfolio['Volatility']*100:.2f}%")
    st.metric("Sharpe Ratio", f"{max_sharpe_portfolio['Sharpe Ratio']:.2f}")
    
    # Extract weights
    ms_weights = [max_sharpe_portfolio[f'Weight_{ticker}'] for ticker in tickers]
    fig_ms = px.pie(names=tickers, values=ms_weights, title="Max Sharpe Allocation Strategy", hole=0.4, template="plotly_dark")
    st.plotly_chart(fig_ms, use_container_width=True)

with m_col2:
    st.write("### 🛡️ Minimum Volatility Portfolio")
    st.metric("Expected Return", f"{min_vol_portfolio['Return']*100:.2f}%")
    st.metric("Annual Volatility", f"{min_vol_portfolio['Volatility']*100:.2f}%")
    st.metric("Sharpe Ratio", f"{min_vol_portfolio['Sharpe Ratio']:.2f}")
    
    # Extract weights
    mv_weights = [min_vol_portfolio[f'Weight_{ticker}'] for ticker in tickers]
    fig_mv = px.pie(names=tickers, values=mv_weights, title="Min Volatility Allocation Strategy", hole=0.4, template="plotly_dark")
    st.plotly_chart(fig_mv, use_container_width=True)

st.divider()

# Fragment optimization isolates visualization settings
@st.fragment
def render_efficient_frontier_view(sim_df, max_s, min_v):
    st.write("### The Efficient Frontier Curve Space")
    
    color_by = st.selectbox("Color Frontier Markers By:", ["Sharpe Ratio", "Return", "Volatility"])
    
    fig = go.Figure()
    
    # Map generated Monte Carlo point clouds
    fig.add_trace(go.Scatter(
        x=sim_df['Volatility'] * 100,
        y=sim_df['Return'] * 100,
        mode='markers',
        marker=dict(
            size=4,
            color=sim_df[color_by],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_by)
        ),
        name='Simulated Portfolios',
        text=[
            "<br>".join([f"{t}: {sim_df.iloc[i][f'Weight_{t}']*100:.1f}%" for t in tickers])
            for i in range(len(sim_df))
        ],
        hovertemplate="<b>Volatility:</b> %{x:.2f}%<br><b>Return:</b> %{y:.2f}%<br><br><b>Weights:</b><br>%{text}<extra></extra>"
    ))
    
    # Explicitly highlight Max Sharpe point target
    fig.add_trace(go.Scatter(
        x=[max_s['Volatility'] * 100], y=[max_s['Return'] * 100],
        mode='markers', marker=dict(color='red', size=14, symbol='star'),
        name='Maximum Sharpe Ratio'
    ))
    
    # Explicitly highlight Min Volatility point target
    fig.add_trace(go.Scatter(
        x=[min_v['Volatility'] * 100], y=[min_v['Return'] * 100],
        mode='markers', marker=dict(color='cyan', size=14, symbol='diamond'),
        name='Minimum Volatility'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Annualized Volatility (Risk %)",
        yaxis_title="Expected Annual Return (%)",
        margin=dict(l=40, r=20, t=20, b=40),
        height=550,
    )
    
    st.plotly_chart(fig, use_container_width=True)

render_efficient_frontier_view(simulated_portfolios, max_sharpe_portfolio, min_vol_portfolio)

st.divider()

# ----------------------------------------------------
# 6. ASSET RELATION MATRIX
# ----------------------------------------------------
st.write("### Underlying Asset Statistics & Correlations")
c_tabs1, c_tabs2 = st.tabs(["Correlation Grid Matrix", "Normalized Historical Performance Trend"])

with c_tabs1:
    corr_matrix = daily_returns.corr()
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="RdBu_r", 
        zmin=-1, zmax=1,
        template="plotly_dark",
        title="Asset Daily Returns Correlation Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with c_tabs2:
    # Normalize price history series to base 100 initialization scale
    normalized_prices = (adj_close_df / adj_close_df.iloc[0]) * 100
    fig_line = px.line(
        normalized_prices, 
        labels={"value": "Normalized Growth (Base 100)", "Date": "Timeline"},
        title="Growth Comparison of $100 Initial Investment Base",
        template="plotly_dark"
    )
    st.plotly_chart(fig_line, use_container_width=True)