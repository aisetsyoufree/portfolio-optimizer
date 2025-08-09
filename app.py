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

class PortfolioOptimizer:
    def __init__(self):
        self.etf_list = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'BND', 'TLT', 'GLD', 'VNQ',
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE',
            'VIG', 'VYM', 'VTV', 'VUG', 'VB', 'VO', 'VXF', 'VXUS', 'VTEB', 'VMOT',
            'EFA', 'EEM', 'AGG', 'LQD', 'HYG', 'IEFA', 'IEMG', 'IJH', 'IJR', 'ITOT'
        ]
        
        self.mutual_funds = [
            'VTSAX', 'VTIAX', 'VBTLX', 'VGTSX', 'VTSMX', 'VFWAX', 'VTABX', 'VGSLX',
            'FXNAX', 'FSKAX', 'FTIHX', 'FXNAX', 'SWPPX', 'SWTSX', 'SWAGX', 'SWISX'
        ]
        
        self.indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }

    def fetch_data(self, symbols, start_date, end_date):
        """Fetch historical data for given symbols"""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                return pd.DataFrame()
            
            # Handle single symbol case
            if len(symbols) == 1:
                if 'Adj Close' in data.columns:
                    return data[['Adj Close']].rename(columns={'Adj Close': symbols[0]})
                else:
                    return data[['Close']].rename(columns={'Close': symbols[0]})
            
            # Handle multiple symbols case
            if 'Adj Close' in data.columns:
                return data['Adj Close']
            else:
                return data['Close']
                
        except Exception as e:
            st.error(f"Error fetching data for {symbols}: {str(e)}")
            return pd.DataFrame()

    def calculate_returns(self, prices):
        """Calculate daily returns"""
        return prices.pct_change().dropna()

    def get_market_regime(self, index_returns, date):
        """Determine if market was up or down around the selected date"""
        # Look at 30-day window around the date
        start_window = date - timedelta(days=15)
        end_window = date + timedelta(days=15)
        
        window_returns = index_returns.loc[start_window:end_window]
        avg_return = window_returns.mean()
        
        return "Up Market" if avg_return > 0 else "Down Market"

    def optimize_portfolio(self, returns, index_returns, target_date, num_assets=5):
        """Optimize portfolio to beat index in both up and down markets"""
        
        # Convert target_date to pandas Timestamp for proper indexing
        target_date = pd.Timestamp(target_date)
        
        # Filter data around target date
        start_analysis = target_date - timedelta(days=252)  # 1 year of data
        end_analysis = target_date + timedelta(days=30)
        
        # Ensure we have overlapping date ranges
        common_dates = returns.index.intersection(index_returns.index)
        if len(common_dates) == 0:
            st.error("No overlapping dates between fund data and index data")
            return pd.DataFrame(), []
        
        # Filter to common date range
        start_date = max(start_analysis, common_dates.min())
        end_date = min(end_analysis, common_dates.max())
        
        returns_period = returns.loc[start_date:end_date]
        index_period = index_returns.loc[start_date:end_date]
        
        # Remove assets with insufficient data
        valid_assets = returns_period.columns[returns_period.count() > 200]
        returns_clean = returns_period[valid_assets].fillna(0)
        
        if len(returns_clean.columns) < num_assets:
            num_assets = len(returns_clean.columns)
        
        # Separate up and down market periods
        up_market_mask = index_period > 0
        down_market_mask = index_period <= 0
        
        up_returns = returns_clean[up_market_mask].mean()
        down_returns = returns_clean[down_market_mask].mean()
        index_up = index_period[up_market_mask].mean()
        index_down = index_period[down_market_mask].mean()
        
        # Calculate correlation matrix
        corr_matrix = returns_clean.corr()
        
        # Objective function: maximize outperformance in both regimes
        def objective(weights):
            portfolio_up = np.dot(weights, up_returns)
            portfolio_down = np.dot(weights, down_returns)
            
            # Penalty for high correlation (diversification)
            portfolio_var = np.dot(weights, np.dot(corr_matrix.loc[selected_assets, selected_assets], weights))
            
            # Maximize outperformance minus variance penalty
            up_outperform = portfolio_up - index_up
            down_outperform = portfolio_down - index_down
            
            return -(up_outperform + down_outperform - 0.1 * portfolio_var)
        
        # Select top performing assets for optimization
        combined_score = (up_returns - index_up) + (down_returns - index_down)
        selected_assets = combined_score.nlargest(min(num_assets * 2, len(combined_score))).index.tolist()
        
        n_assets = len(selected_assets)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 0.4) for _ in range(n_assets))  # Max 40% in any single asset
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            # Get top assets by weight
            weights_df = pd.DataFrame({
                'Asset': selected_assets,
                'Weight': result.x
            }).sort_values('Weight', ascending=False).head(num_assets)
            
            # Normalize weights to sum to 1
            weights_df['Weight'] = weights_df['Weight'] / weights_df['Weight'].sum()
            
            return weights_df, selected_assets
        else:
            # Fallback: equal weight top performers
            top_assets = combined_score.nlargest(num_assets).index.tolist()
            weights_df = pd.DataFrame({
                'Asset': top_assets,
                'Weight': [1/num_assets] * num_assets
            })
            return weights_df, top_assets

def main():
    st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
    
    st.title("🚀 Smart Portfolio Optimizer")
    st.markdown("*Beat the market in both up and down cycles*")
    
    optimizer = PortfolioOptimizer()
    
    # Sidebar inputs
    st.sidebar.header("Configuration")
    
    selected_index = st.sidebar.selectbox(
        "Select Market Index",
        list(optimizer.indices.keys())
    )
    
    target_date = st.sidebar.date_input(
        "Select Target Date",
        value=datetime.now() - timedelta(days=30),
        max_value=datetime.now() - timedelta(days=1)
    )
    
    num_funds = st.sidebar.slider("Number of Funds in Portfolio", 3, 5, 4)
    
    if st.sidebar.button("🔍 Analyze & Optimize", type="primary"):
        with st.spinner("Fetching market data and optimizing portfolio..."):
            
            # Fetch index data
            index_symbol = optimizer.indices[selected_index]
            start_date = target_date - timedelta(days=400)
            end_date = target_date + timedelta(days=60)
            
            # Get index returns
            index_data = optimizer.fetch_data(index_symbol, start_date, end_date)
            if index_data.empty:
                st.error("Failed to fetch index data")
                return
                
            index_returns = optimizer.calculate_returns(index_data)
            
            # Ensure we have the correct column name
            if len(index_returns.columns) > 0:
                index_col = index_returns.columns[0]
            else:
                st.error("No valid index data found")
                return
            
            # Get market regime
            market_regime = optimizer.get_market_regime(index_returns[index_col], target_date)
            
            # Fetch ETF and mutual fund data
            all_symbols = optimizer.etf_list + optimizer.mutual_funds
            
            st.info(f"Analyzing {len(all_symbols)} ETFs and mutual funds...")
            
            # Fetch data in batches to avoid API limits
            batch_size = 20
            all_returns = pd.DataFrame()
            
            for i in range(0, len(all_symbols), batch_size):
                batch = all_symbols[i:i+batch_size]
                batch_data = optimizer.fetch_data(batch, start_date, end_date)
                if not batch_data.empty:
                    batch_returns = optimizer.calculate_returns(batch_data)
                    all_returns = pd.concat([all_returns, batch_returns], axis=1)
            
            if all_returns.empty:
                st.error("Failed to fetch fund data")
                return
            
            # Optimize portfolio
            optimal_weights, selected_assets = optimizer.optimize_portfolio(
                all_returns, index_returns[index_col], target_date, num_funds
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📊 Market Analysis for {target_date}")
                st.metric("Selected Index", selected_index)
                st.metric("Market Regime", market_regime)
                
                # Show index performance
                try:
                    date_range = index_returns[index_col].loc[target_date:target_date+timedelta(days=1)]
                    index_return_on_date = date_range.iloc[0] if len(date_range) > 0 else 0
                except:
                    index_return_on_date = 0
                st.metric("Index Return on Date", f"{index_return_on_date:.2%}")
            
            with col2:
                st.subheader("🎯 Optimized Portfolio")
                
                # Display portfolio weights
                fig = px.pie(optimal_weights, values='Weight', names='Asset', 
                           title="Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio details table
            st.subheader("📈 Portfolio Details")
            
            # Calculate expected performance
            portfolio_returns = all_returns[optimal_weights['Asset']].fillna(0)
            weights_array = optimal_weights['Weight'].values
            
            # Performance metrics
            portfolio_daily_returns = (portfolio_returns * weights_array).sum(axis=1)
            
            # Up/down market performance
            up_mask = index_returns[index_col] > 0
            down_mask = index_returns[index_col] <= 0
            
            portfolio_up_perf = portfolio_daily_returns[up_mask].mean()
            portfolio_down_perf = portfolio_daily_returns[down_mask].mean()
            index_up_perf = index_returns[index_col][up_mask].mean()
            index_down_perf = index_returns[index_col][down_mask].mean()
            
            # Enhanced portfolio table
            portfolio_details = optimal_weights.copy()
            portfolio_details['Weight'] = portfolio_details['Weight'].apply(lambda x: f"{x:.1%}")
            
            # Add recent performance for each asset
            recent_performance = []
            for asset in optimal_weights['Asset']:
                if asset in all_returns.columns:
                    recent_return = all_returns[asset].loc[target_date-timedelta(days=30):target_date].mean()
                    recent_performance.append(f"{recent_return:.2%}")
                else:
                    recent_performance.append("N/A")
            
            portfolio_details['30-Day Avg Return'] = recent_performance
            
            st.dataframe(portfolio_details, use_container_width=True)
            
            # Performance comparison
            st.subheader("⚡ Performance Comparison")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Portfolio Up Market", 
                    f"{portfolio_up_perf:.2%}",
                    delta=f"{(portfolio_up_perf - index_up_perf):.2%}"
                )
            
            with col2:
                st.metric(
                    "Index Up Market", 
                    f"{index_up_perf:.2%}"
                )
            
            with col3:
                st.metric(
                    "Portfolio Down Market", 
                    f"{portfolio_down_perf:.2%}",
                    delta=f"{(portfolio_down_perf - index_down_perf):.2%}"
                )
            
            with col4:
                st.metric(
                    "Index Down Market", 
                    f"{index_down_perf:.2%}"
                )
            
            # Performance chart
            st.subheader("📊 Historical Performance Comparison")
            
            # Calculate cumulative returns
            portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
            index_cumulative = (1 + index_returns[index_col]).cumprod()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_cumulative.index,
                y=portfolio_cumulative.values,
                mode='lines',
                name='Optimized Portfolio',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=index_cumulative.index,
                y=index_cumulative.values,
                mode='lines',
                name=selected_index,
                line=dict(color='blue', width=2)
            ))
            
            # Add vertical line for target date
            fig.add_vline(x=target_date, line_dash="dash", line_color="red", 
                         annotation_text="Target Date")
            
            fig.update_layout(
                title="Cumulative Returns Comparison",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            st.subheader("🛡️ Risk Analysis")
            
            portfolio_vol = portfolio_daily_returns.std() * np.sqrt(252)
            index_vol = index_returns[index_col].std() * np.sqrt(252)
            portfolio_sharpe = (portfolio_daily_returns.mean() * 252) / portfolio_vol
            index_sharpe = (index_returns[index_col].mean() * 252) / index_vol
            
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            
            with risk_col1:
                st.metric("Portfolio Volatility", f"{portfolio_vol:.1%}")
            
            with risk_col2:
                st.metric("Index Volatility", f"{index_vol:.1%}")
            
            with risk_col3:
                st.metric("Portfolio Sharpe", f"{portfolio_sharpe:.2f}")
            
            with risk_col4:
                st.metric("Index Sharpe", f"{index_sharpe:.2f}")

if __name__ == "__main__":
    main()
