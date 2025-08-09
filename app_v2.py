import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch data for a single symbol with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if not data.empty and 'Close' in data.columns:
            return data['Close']
        return pd.Series(dtype=float)
    except:
        return pd.Series(dtype=float)

def calculate_returns(prices):
    """Calculate daily returns"""
    return prices.pct_change().dropna()

def main():
    st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
    
    st.title("🚀 Smart Portfolio Optimizer")
    st.markdown("*Beat the market in both up and down cycles*")
    
    # Define available indices
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC', 
        'Dow Jones': '^DJI',
        'Russell 2000': '^RUT'
    }
    
    # Popular ETFs and mutual funds
    popular_funds = [
        'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'BND', 'TLT', 'GLD', 'VNQ',
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'VIG', 'VYM', 'VTV', 'VUG',
        'VTSAX', 'VTIAX', 'VBTLX', 'FXNAX', 'FSKAX', 'SWPPX', 'SWTSX'
    ]
    
    # Sidebar inputs
    st.sidebar.header("📋 Configuration")
    
    selected_index = st.sidebar.selectbox(
        "Select Market Index",
        list(indices.keys())
    )
    
    target_date = st.sidebar.date_input(
        "Select Target Date",
        value=datetime.now() - timedelta(days=30),
        max_value=datetime.now() - timedelta(days=1)
    )
    
    num_funds = st.sidebar.slider("Number of Funds in Portfolio", 3, 5, 4)
    
    # Requirements clarification section
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Strategy Options")
    
    strategy_focus = st.sidebar.radio(
        "Optimization Focus",
        ["Balanced (Up & Down Markets)", "Downside Protection", "Growth Focus"]
    )
    
    fund_universe = st.sidebar.selectbox(
        "Fund Universe",
        ["Popular ETFs & Mutual Funds (~27)", "Extended Universe (~50+)", "Custom Selection"]
    )
    
    time_window = st.sidebar.selectbox(
        "Analysis Window",
        ["1 Year Before Target Date", "6 Months Before Target Date", "2 Years Before Target Date"]
    )
    
    if st.sidebar.button("🔍 Analyze & Optimize", type="primary"):
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch index data
            status_text.text("📊 Fetching index data...")
            progress_bar.progress(20)
            
            index_symbol = indices[selected_index]
            
            # Set date ranges based on user selection
            days_back = {"1 Year Before Target Date": 365, 
                        "6 Months Before Target Date": 180, 
                        "2 Years Before Target Date": 730}[time_window]
            
            start_date = target_date - timedelta(days=days_back)
            end_date = target_date + timedelta(days=30)
            
            # Fetch index data
            index_data = fetch_stock_data(index_symbol, start_date, end_date)
            
            if index_data.empty:
                st.error(f"❌ Failed to fetch data for {selected_index}. Please try a different date range.")
                return
            
            index_returns = calculate_returns(index_data)
            
            # Step 2: Determine market regime
            status_text.text("🔍 Analyzing market conditions...")
            progress_bar.progress(40)
            
            # Look at 30-day window around target date
            window_start = target_date - timedelta(days=15)
            window_end = target_date + timedelta(days=15)
            
            try:
                window_returns = index_returns.loc[window_start:window_end]
                avg_return = window_returns.mean()
                market_regime = "📈 Up Market" if avg_return > 0 else "📉 Down Market"
            except:
                market_regime = "❓ Unknown"
            
            # Step 3: Fetch fund data
            status_text.text("💼 Fetching fund data...")
            progress_bar.progress(60)
            
            # Determine fund list based on user selection
            if fund_universe == "Popular ETFs & Mutual Funds (~27)":
                fund_list = popular_funds
            else:
                fund_list = popular_funds  # For now, use same list
            
            # Fetch fund data
            fund_data = {}
            valid_funds = []
            
            for i, fund in enumerate(fund_list):
                try:
                    data = fetch_stock_data(fund, start_date, end_date)
                    if not data.empty and len(data) > 50:  # Minimum data requirement
                        fund_data[fund] = data
                        valid_funds.append(fund)
                except:
                    continue
                
                # Update progress
                progress_bar.progress(60 + (i / len(fund_list)) * 20)
            
            if len(valid_funds) < num_funds:
                st.warning(f"⚠️ Only found {len(valid_funds)} valid funds. Adjusting portfolio size.")
                num_funds = min(num_funds, len(valid_funds))
            
            # Step 4: Calculate returns and optimize
            status_text.text("🎯 Optimizing portfolio...")
            progress_bar.progress(85)
            
            # Create returns dataframe
            fund_returns = pd.DataFrame()
            for fund, prices in fund_data.items():
                returns = calculate_returns(prices)
                fund_returns[fund] = returns
            
            # Align dates
            common_dates = fund_returns.index.intersection(index_returns.index)
            fund_returns_aligned = fund_returns.loc[common_dates]
            index_returns_aligned = index_returns.loc[common_dates]
            
            # Simple optimization: select top performers with good diversification
            # Calculate performance metrics for each fund
            fund_metrics = {}
            
            for fund in fund_returns_aligned.columns:
                fund_ret = fund_returns_aligned[fund].dropna()
                if len(fund_ret) > 30:
                    # Up market performance
                    up_mask = index_returns_aligned > 0
                    up_perf = fund_ret[up_mask].mean() if up_mask.sum() > 0 else 0
                    
                    # Down market performance  
                    down_mask = index_returns_aligned <= 0
                    down_perf = fund_ret[down_mask].mean() if down_mask.sum() > 0 else 0
                    
                    # Overall metrics
                    total_return = fund_ret.mean()
                    volatility = fund_ret.std()
                    sharpe = total_return / volatility if volatility > 0 else 0
                    
                    # Combined score based on strategy
                    if strategy_focus == "Downside Protection":
                        score = down_perf * 0.6 + up_perf * 0.2 + sharpe * 0.2
                    elif strategy_focus == "Growth Focus":
                        score = up_perf * 0.6 + total_return * 0.3 + sharpe * 0.1
                    else:  # Balanced
                        score = (up_perf + down_perf) * 0.4 + sharpe * 0.2
                    
                    fund_metrics[fund] = {
                        'score': score,
                        'up_perf': up_perf,
                        'down_perf': down_perf,
                        'total_return': total_return,
                        'volatility': volatility,
                        'sharpe': sharpe
                    }
            
            # Select top funds
            sorted_funds = sorted(fund_metrics.items(), key=lambda x: x[1]['score'], reverse=True)
            selected_funds = [fund for fund, _ in sorted_funds[:num_funds]]
            
            # Equal weight allocation (can be enhanced with optimization)
            weights = [1/num_funds] * num_funds
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.success("🎉 Portfolio optimization completed successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📊 Market Analysis")
                st.metric("Selected Index", selected_index)
                st.metric("Market Regime", market_regime)
                st.metric("Analysis Period", f"{days_back} days")
                st.metric("Valid Funds Found", len(valid_funds))
            
            with col2:
                st.subheader("🎯 Optimized Portfolio")
                
                # Create portfolio dataframe
                portfolio_df = pd.DataFrame({
                    'Fund': selected_funds,
                    'Weight': [f"{w:.1%}" for w in weights],
                    'Score': [f"{fund_metrics[fund]['score']:.4f}" for fund in selected_funds]
                })
                
                # Display portfolio allocation
                fig = px.pie(
                    values=weights, 
                    names=selected_funds, 
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio details
            st.subheader("📈 Portfolio Details")
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Performance comparison
            st.subheader("⚡ Performance Analysis")
            
            # Calculate portfolio performance
            portfolio_returns = fund_returns_aligned[selected_funds].fillna(0)
            portfolio_daily_returns = (portfolio_returns * weights).sum(axis=1)
            
            # Up/down market performance
            up_mask = index_returns_aligned > 0
            down_mask = index_returns_aligned <= 0
            
            portfolio_up = portfolio_daily_returns[up_mask].mean() if up_mask.sum() > 0 else 0
            portfolio_down = portfolio_daily_returns[down_mask].mean() if down_mask.sum() > 0 else 0
            index_up = index_returns_aligned[up_mask].mean() if up_mask.sum() > 0 else 0
            index_down = index_returns_aligned[down_mask].mean() if down_mask.sum() > 0 else 0
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            with perf_col1:
                st.metric(
                    "Portfolio Up Market", 
                    f"{portfolio_up:.2%}",
                    delta=f"{(portfolio_up - index_up):.2%}" if index_up != 0 else None
                )
            
            with perf_col2:
                st.metric("Index Up Market", f"{index_up:.2%}")
            
            with perf_col3:
                st.metric(
                    "Portfolio Down Market", 
                    f"{portfolio_down:.2%}",
                    delta=f"{(portfolio_down - index_down):.2%}" if index_down != 0 else None
                )
            
            with perf_col4:
                st.metric("Index Down Market", f"{index_down:.2%}")
            
            # Show individual fund performance
            st.subheader("🔍 Individual Fund Analysis")
            
            fund_analysis = []
            for fund in selected_funds:
                metrics = fund_metrics[fund]
                fund_analysis.append({
                    'Fund': fund,
                    'Up Market Return': f"{metrics['up_perf']:.2%}",
                    'Down Market Return': f"{metrics['down_perf']:.2%}",
                    'Average Return': f"{metrics['total_return']:.2%}",
                    'Volatility': f"{metrics['volatility']:.2%}",
                    'Sharpe Ratio': f"{metrics['sharpe']:.2f}"
                })
            
            st.dataframe(pd.DataFrame(fund_analysis), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 Try selecting a different date or index")

if __name__ == "__main__":
    main()
