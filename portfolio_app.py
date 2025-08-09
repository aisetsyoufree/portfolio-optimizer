import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import requests
import io
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self):
        # Comprehensive list of US equity ETFs and mutual funds
        self.equity_etfs = [
            # Broad Market ETFs
            'SPY', 'IVV', 'VOO', 'VTI', 'ITOT', 'SPTM', 'SCHB',
            # Large Cap
            'QQQ', 'VUG', 'VTV', 'IWF', 'IWD', 'MTUM', 'QUAL', 'USMV',
            # Mid Cap
            'MDY', 'IJH', 'VO', 'SCHM', 'IWR', 'IWP', 'IWS',
            # Small Cap
            'IWM', 'VB', 'IJR', 'SCHA', 'IWN', 'IWO', 'VBR', 'VBK',
            # Sector ETFs
            'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC',
            'VGT', 'VFH', 'VHT', 'VDE', 'VIS', 'VDC', 'VCR', 'VPU', 'VAW', 'VNQ',
            # Factor ETFs
            'VIG', 'VYM', 'DGRO', 'NOBL', 'SCHD', 'DVY', 'HDV',
            # International Equity (US-traded)
            'EFA', 'EEM', 'VEA', 'VWO', 'IEFA', 'IEMG', 'ACWI', 'VXUS'
        ]
        
        self.equity_mutual_funds = [
            # Vanguard
            'VTSAX', 'VTIAX', 'VGTSX', 'VTSMX', 'VFWAX', 'VIGAX', 'VGSLX',
            'VMGMX', 'VSGAX', 'VTMGX', 'VSGDX', 'VEXAX', 'VTRIX', 'VWIGX',
            # Fidelity
            'FXNAX', 'FSKAX', 'FTIHX', 'FZROX', 'FZILX', 'FDGRX', 'FXAIX',
            'FNCMX', 'FREL', 'FSELX', 'FTEC', 'FBIOX', 'FDVV', 'FGDAX',
            # Schwab
            'SWPPX', 'SWTSX', 'SWAGX', 'SWISX', 'SWLGX', 'SWMCX', 'SWSCX',
            'SWTGX', 'SWVGX', 'SWDGX', 'SWYGX', 'SWLVX',
            # T. Rowe Price
            'TRBCX', 'PRGFX', 'PRGTX', 'PRDGX', 'PRNHX', 'PRWCX'
        ]
        
        self.indices = {
            'S&P 500': '^GSPC',
            'NASDAQ Composite': '^IXIC', 
            'Dow Jones Industrial': '^DJI',
            'Russell 2000': '^RUT',
            'Russell 1000': '^RUI',
            'NASDAQ 100': '^NDX'
        }

    def fetch_single_stock(self, symbol, start_date, end_date):
        """Fetch data for a single symbol with robust error handling"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
                
            # Use Close price if available
            if 'Close' in hist.columns and len(hist) > 20:  # Minimum 20 days of data
                return hist['Close']
            return None
            
        except Exception as e:
            return None

    def fetch_fund_universe(self, start_date, end_date, max_funds=100):
        """Fetch data for equity funds with progress tracking"""
        all_symbols = self.equity_etfs + self.equity_mutual_funds
        
        # Shuffle for better distribution of successful fetches
        np.random.shuffle(all_symbols)
        
        fund_data = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(all_symbols[:max_funds]):
            status_text.text(f"Fetching {symbol}... ({i+1}/{min(len(all_symbols), max_funds)})")
            
            data = self.fetch_single_stock(symbol, start_date, end_date)
            if data is not None:
                fund_data[symbol] = data
            
            progress_bar.progress((i + 1) / min(len(all_symbols), max_funds))
            
            # Stop if we have enough funds
            if len(fund_data) >= max_funds:
                break
        
        progress_bar.empty()
        status_text.empty()
        
        return fund_data

    def calculate_performance_metrics(self, returns, index_returns, strategy_focus):
        """Calculate performance metrics based on strategy focus"""
        
        # Separate up and down market periods
        up_mask = index_returns > 0
        down_mask = index_returns <= 0
        
        metrics = {}
        
        for fund in returns.columns:
            fund_returns = returns[fund].dropna()
            
            if len(fund_returns) < 30:  # Skip funds with insufficient data
                continue
                
            # Align with index dates
            common_dates = fund_returns.index.intersection(index_returns.index)
            if len(common_dates) < 30:
                continue
                
            fund_aligned = fund_returns.loc[common_dates]
            index_aligned = index_returns.loc[common_dates]
            
            # Performance in different market conditions
            up_perf = fund_aligned[index_aligned > 0].mean() if (index_aligned > 0).sum() > 0 else 0
            down_perf = fund_aligned[index_aligned <= 0].mean() if (index_aligned <= 0).sum() > 0 else 0
            
            # Risk metrics
            total_return = fund_aligned.mean()
            volatility = fund_aligned.std()
            sharpe = (total_return / volatility) if volatility > 0 else 0
            
            # Index comparison
            index_up = index_aligned[index_aligned > 0].mean() if (index_aligned > 0).sum() > 0 else 0
            index_down = index_aligned[index_aligned <= 0].mean() if (index_aligned <= 0).sum() > 0 else 0
            
            up_alpha = up_perf - index_up
            down_alpha = down_perf - index_down
            
            # Strategy-based scoring
            if strategy_focus == "Downside Protection":
                score = down_alpha * 0.5 + up_alpha * 0.2 + sharpe * 0.3
            elif strategy_focus == "Growth Focus":
                score = up_alpha * 0.5 + total_return * 0.3 + sharpe * 0.2
            else:  # Balanced
                score = (up_alpha + down_alpha) * 0.4 + sharpe * 0.3 + total_return * 0.3
            
            metrics[fund] = {
                'score': score,
                'up_performance': up_perf,
                'down_performance': down_perf,
                'up_alpha': up_alpha,
                'down_alpha': down_alpha,
                'total_return': total_return,
                'volatility': volatility,
                'sharpe': sharpe
            }
        
        return metrics

    def optimize_weights(self, selected_funds, fund_returns, index_returns, strategy_focus):
        """Optimize portfolio weights using mathematical optimization"""
        
        if len(selected_funds) < 2:
            return [1.0] * len(selected_funds)
        
        # Prepare data for optimization
        returns_matrix = fund_returns[selected_funds].fillna(0)
        
        # Align dates
        common_dates = returns_matrix.index.intersection(index_returns.index)
        returns_aligned = returns_matrix.loc[common_dates]
        index_aligned = index_returns.loc[common_dates]
        
        if len(returns_aligned) < 30:
            return [1/len(selected_funds)] * len(selected_funds)
        
        # Calculate expected returns and covariance
        mean_returns = returns_aligned.mean()
        cov_matrix = returns_aligned.cov()
        
        # Up/down market returns
        up_mask = index_aligned > 0
        down_mask = index_aligned <= 0
        
        up_returns = returns_aligned[up_mask].mean() if up_mask.sum() > 0 else mean_returns
        down_returns = returns_aligned[down_mask].mean() if down_mask.sum() > 0 else mean_returns
        
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            portfolio_up = np.dot(weights, up_returns)
            portfolio_down = np.dot(weights, down_returns)
            
            if strategy_focus == "Downside Protection":
                return -(portfolio_down * 0.6 + portfolio_return * 0.3 - portfolio_vol * 0.1)
            elif strategy_focus == "Growth Focus":
                return -(portfolio_up * 0.6 + portfolio_return * 0.3 - portfolio_vol * 0.1)
            else:  # Balanced
                return -((portfolio_up + portfolio_down) * 0.4 + portfolio_return * 0.4 - portfolio_vol * 0.2)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.05, 0.4) for _ in range(len(selected_funds)))  # 5% min, 40% max
        initial_guess = np.array([1/len(selected_funds)] * len(selected_funds))
        
        try:
            result = minimize(objective, initial_guess, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x.tolist()
            else:
                return [1/len(selected_funds)] * len(selected_funds)
        except:
            return [1/len(selected_funds)] * len(selected_funds)

def create_excel_export(portfolio_df, fund_performance_summary, detailed_analysis, 
                       index_data, fund_returns, selected_funds, optimal_weights):
    """Create Excel file with all portfolio data"""
    
    # Create Excel buffer
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Portfolio summary
        portfolio_df.to_excel(writer, sheet_name='Portfolio_Summary', index=False)
        
        # Individual fund performance
        if fund_performance_summary:
            pd.DataFrame(fund_performance_summary).to_excel(writer, sheet_name='Fund_Performance', index=False)
        
        # Detailed analysis
        if detailed_analysis:
            pd.DataFrame(detailed_analysis).to_excel(writer, sheet_name='Detailed_Analysis', index=False)
        
        # Raw returns data
        if not fund_returns.empty:
            fund_returns_export = fund_returns[selected_funds].copy()
            # Remove timezone from index if present
            if hasattr(fund_returns_export.index, 'tz') and fund_returns_export.index.tz is not None:
                fund_returns_export.index = fund_returns_export.index.tz_localize(None)
            fund_returns_export.to_excel(writer, sheet_name='Fund_Returns')
        
        # Index data
        if index_data is not None:
            index_df = pd.DataFrame({'Index_Price': index_data, 'Index_Returns': index_data.pct_change()})
            # Remove timezone from index if present
            if hasattr(index_df.index, 'tz') and index_df.index.tz is not None:
                index_df.index = index_df.index.tz_localize(None)
            index_df.to_excel(writer, sheet_name='Index_Data')
        
        # Portfolio weights
        weights_df = pd.DataFrame({
            'Fund': selected_funds,
            'Weight': optimal_weights,
            'Weight_Percentage': [f"{w:.2%}" for w in optimal_weights]
        })
        weights_df.to_excel(writer, sheet_name='Portfolio_Weights', index=False)
    
    return buffer.getvalue()

def main():
    st.set_page_config(page_title="Smart Portfolio Optimizer", layout="wide")
    
    st.title("🚀 Smart Portfolio Optimizer")
    st.markdown("*Outperform market indices with optimized equity fund portfolios*")
    
    optimizer = PortfolioOptimizer()
    
    # Sidebar configuration
    st.sidebar.header("📊 Market Analysis")
    
    selected_index = st.sidebar.selectbox(
        "Select Benchmark Index",
        list(optimizer.indices.keys())
    )
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.sidebar.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now() - timedelta(days=30)
        )
    
    with col2:
        end_date = st.sidebar.date_input(
            "End Date", 
            value=datetime.now() - timedelta(days=1),
            max_value=datetime.now() - timedelta(days=1)
        )
    
    # Validate date range
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        return
    
    if (end_date - start_date).days < 90:
        st.sidebar.warning("Consider using at least 3 months of data for better analysis")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Portfolio Strategy")
    
    strategy_focus = st.sidebar.radio(
        "Optimization Strategy",
        ["Balanced (Equal Emphasis)", "Downside Protection", "Growth Focus"],
        help="Balanced: Equal weight on up/down markets | Downside: Focus on bear market protection | Growth: Focus on bull market performance"
    )
    
    num_funds = st.sidebar.slider("Portfolio Size", 3, 8, 5)
    max_funds_analyze = st.sidebar.slider("Max Funds to Analyze", 50, 200, 100, 
                                         help="Higher numbers = more comprehensive but slower")
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        min_data_days = st.slider("Minimum Data Days", 30, 180, 60)
        max_weight = st.slider("Max Weight per Fund", 0.2, 0.5, 0.35)
        include_international = st.checkbox("Include International Funds", value=True)
    
    # Initialize session state
    if 'optimization_complete' not in st.session_state:
        st.session_state.optimization_complete = False
    
    if st.sidebar.button("🔍 Optimize Portfolio", type="primary"):
        # Clear previous results
        st.session_state.optimization_complete = False
        
        if not include_international:
            # Filter out international funds
            optimizer.equity_etfs = [etf for etf in optimizer.equity_etfs 
                                   if etf not in ['EFA', 'EEM', 'VEA', 'VWO', 'IEFA', 'IEMG', 'ACWI', 'VXUS']]
        
        with st.spinner("🔄 Starting portfolio optimization..."):
            
            # Step 1: Fetch index data
            st.info("📊 Fetching benchmark index data...")
            
            index_symbol = optimizer.indices[selected_index]
            index_data = optimizer.fetch_single_stock(index_symbol, start_date, end_date)
            
            if index_data is None or len(index_data) < min_data_days:
                st.error(f"❌ Failed to fetch sufficient data for {selected_index}")
                st.info("💡 Try selecting a different date range or index")
                return
            
            index_returns = index_data.pct_change().dropna()
            
            # Market regime analysis
            up_days = (index_returns > 0).sum()
            down_days = (index_returns <= 0).sum()
            avg_up_return = index_returns[index_returns > 0].mean() if up_days > 0 else 0
            avg_down_return = index_returns[index_returns <= 0].mean() if down_days > 0 else 0
            
            # Step 2: Fetch fund universe
            st.info(f"💼 Analyzing up to {max_funds_analyze} equity funds...")
            
            fund_data = optimizer.fetch_fund_universe(start_date, end_date, max_funds_analyze)
            
            if len(fund_data) < num_funds:
                st.error(f"❌ Only found {len(fund_data)} valid funds, need at least {num_funds}")
                st.info("💡 Try expanding the date range or reducing portfolio size")
                return
            
            st.success(f"✅ Successfully fetched data for {len(fund_data)} funds")
            
            # Step 3: Calculate returns
            st.info("📈 Calculating fund returns and performance metrics...")
            
            fund_returns = pd.DataFrame()
            for symbol, prices in fund_data.items():
                returns = prices.pct_change().dropna()
                if len(returns) >= min_data_days:
                    fund_returns[symbol] = returns
            
            # Step 4: Calculate performance metrics
            metrics = optimizer.calculate_performance_metrics(
                fund_returns, index_returns, strategy_focus
            )
            
            if len(metrics) < num_funds:
                st.error(f"❌ Insufficient funds with good data quality")
                return
            
            # Step 5: Select top funds
            sorted_funds = sorted(metrics.items(), key=lambda x: x[1]['score'], reverse=True)
            selected_funds = [fund for fund, _ in sorted_funds[:num_funds]]
            
            # Step 6: Optimize weights
            st.info("🎯 Optimizing portfolio weights...")
            
            optimal_weights = optimizer.optimize_weights(
                selected_funds, fund_returns, index_returns, strategy_focus
            )
            
            # Store results in session state
            st.session_state.optimization_complete = True
            st.session_state.selected_funds = selected_funds
            st.session_state.optimal_weights = optimal_weights
            st.session_state.metrics = metrics
            st.session_state.fund_returns = fund_returns
            st.session_state.index_returns = index_returns
            st.session_state.portfolio_daily_returns = (fund_returns[selected_funds].fillna(0) * optimal_weights).sum(axis=1)
            st.session_state.index_data = index_data
            st.session_state.selected_index = selected_index
            st.session_state.up_days = up_days
            st.session_state.down_days = down_days
            st.session_state.avg_up_return = avg_up_return
            st.session_state.avg_down_return = avg_down_return
            st.session_state.fund_data = fund_data  # Store raw price data
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
            
    # Display results if optimization has been completed
    if st.session_state.optimization_complete:
        # Get data from session state
        selected_funds = st.session_state.selected_funds
        optimal_weights = st.session_state.optimal_weights
        metrics = st.session_state.metrics
        fund_returns = st.session_state.fund_returns
        index_returns = st.session_state.index_returns
        portfolio_daily_returns = st.session_state.portfolio_daily_returns
        index_data = st.session_state.index_data
        selected_index = st.session_state.selected_index
        up_days = st.session_state.up_days
        down_days = st.session_state.down_days
        avg_up_return = st.session_state.avg_up_return
        avg_down_return = st.session_state.avg_down_return
        fund_data = st.session_state.fund_data
        start_date = st.session_state.start_date
        end_date = st.session_state.end_date
        
        # Results Display
        st.success("🎉 Portfolio optimization completed!")
        
        # Market Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Market Analysis")
            st.metric("Benchmark Index", selected_index)
            st.metric("Analysis Period", f"{(end_date - start_date).days} days")
            st.metric("Up Market Days", f"{up_days} ({up_days/(up_days+down_days)*100:.1f}%)")
            st.metric("Down Market Days", f"{down_days} ({down_days/(up_days+down_days)*100:.1f}%)")
        
        with col2:
            st.subheader("📈 Index Performance")
            total_index_return = (index_data.iloc[-1] / index_data.iloc[0] - 1)
            st.metric("Total Index Return", f"{total_index_return:.2%}")
            st.metric("Avg Up Market Return", f"{avg_up_return:.2%}")
            st.metric("Avg Down Market Return", f"{avg_down_return:.2%}")
            st.metric("Index Volatility", f"{index_returns.std() * np.sqrt(252):.1%}")
        
        # Portfolio Allocation
        st.subheader("🎯 Optimized Portfolio")
        
        portfolio_df = pd.DataFrame({
            'Fund': selected_funds,
            'Weight': [f"{w:.1%}" for w in optimal_weights],
            'Score': [f"{metrics[fund]['score']:.4f}" for fund in selected_funds],
            'Up Market Alpha': [f"{metrics[fund]['up_alpha']:.2%}" for fund in selected_funds],
            'Down Market Alpha': [f"{metrics[fund]['down_alpha']:.2%}" for fund in selected_funds]
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.pie(values=optimal_weights, names=selected_funds, 
                       title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Export and raw data options
            col1, col2 = st.columns(2)
            with col1:
                # Excel export button
                fund_performance_summary = []
                for i, fund in enumerate(selected_funds):
                    fund_returns_series = fund_returns[fund].fillna(0)
                    fund_cumulative = (1 + fund_returns_series).cumprod()
                    
                    total_return = (fund_cumulative.iloc[-1] - 1) * 100 if len(fund_cumulative) > 0 else 0
                    weight = optimal_weights[i]
                    contribution = total_return * weight
                    
                    fund_performance_summary.append({
                        'Fund': fund,
                        'Weight': f"{weight:.1%}",
                        'Total Return': f"{total_return:.2f}%",
                        'Contribution to Portfolio': f"{contribution:.2f}%",
                        'Up Market Alpha': f"{metrics[fund]['up_alpha']:.2%}",
                        'Down Market Alpha': f"{metrics[fund]['down_alpha']:.2%}"
                    })
                
                detailed_analysis = []
                for fund in selected_funds:
                    m = metrics[fund]
                    detailed_analysis.append({
                        'Fund': fund,
                        'Weight': f"{optimal_weights[selected_funds.index(fund)]:.1%}",
                        'Up Market Return': f"{m['up_performance']:.2%}",
                        'Down Market Return': f"{m['down_performance']:.2%}",
                        'Up Market Alpha': f"{m['up_alpha']:.2%}",
                        'Down Market Alpha': f"{m['down_alpha']:.2%}",
                        'Volatility': f"{m['volatility']:.2%}",
                        'Sharpe Ratio': f"{m['sharpe']:.2f}",
                        'Score': f"{m['score']:.4f}"
                    })
                
                excel_data = create_excel_export(
                    portfolio_df, fund_performance_summary, detailed_analysis, 
                    index_data, fund_returns, selected_funds, optimal_weights
                )
                st.download_button(
                    label="📊 Export to Excel",
                    data=excel_data,
                    file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # Raw data toggle
                show_raw_data = st.checkbox("📋 Show Raw Calculation Data", value=False,
                                          help="Display all raw returns data used in calculations")
        
        # Performance Analysis
        st.subheader("⚡ Performance Comparison")
        
        # Performance metrics
        portfolio_up_perf = portfolio_daily_returns[index_returns > 0].mean() if (index_returns > 0).sum() > 0 else 0
        portfolio_down_perf = portfolio_daily_returns[index_returns <= 0].mean() if (index_returns <= 0).sum() > 0 else 0
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric(
                "Portfolio Up Market", 
                f"{portfolio_up_perf:.2%}",
                delta=f"{(portfolio_up_perf - avg_up_return):.2%}"
            )
        
        with perf_col2:
            st.metric("Index Up Market", f"{avg_up_return:.2%}")
        
        with perf_col3:
            st.metric(
                "Portfolio Down Market", 
                f"{portfolio_down_perf:.2%}",
                delta=f"{(portfolio_down_perf - avg_down_return):.2%}"
            )
        
        with perf_col4:
            st.metric("Index Down Market", f"{avg_down_return:.2%}")
        
        # Cumulative performance chart
        st.subheader("📊 Cumulative Performance")
        
        # Toggle for individual fund performance
        show_individual_funds = st.checkbox("📈 Show Individual Fund Performance", value=False,
                                           help="Toggle to display each fund's performance on the chart")
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
        index_cumulative = (1 + index_returns).cumprod()
        
        fig = go.Figure()
        
        # Add portfolio and index lines (always shown)
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=(portfolio_cumulative - 1) * 100,
            mode='lines',
            name='🎯 Optimized Portfolio',
            line=dict(color='green', width=3),
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=index_cumulative.index,
            y=(index_cumulative - 1) * 100,
            mode='lines',
            name=f'📊 {selected_index}',
            line=dict(color='blue', width=3),
            hovertemplate=f'<b>{selected_index}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
        ))
        
        # Add individual fund performance if toggled on
        if show_individual_funds:
            colors = px.colors.qualitative.Set3
            for i, fund in enumerate(selected_funds):
                fund_returns_series = fund_returns[fund].fillna(0)
                fund_cumulative = (1 + fund_returns_series).cumprod()
                
                # Align dates with portfolio
                common_dates = fund_cumulative.index.intersection(portfolio_cumulative.index)
                fund_cumulative_aligned = fund_cumulative.loc[common_dates]
                
                fig.add_trace(go.Scatter(
                    x=fund_cumulative_aligned.index,
                    y=(fund_cumulative_aligned - 1) * 100,
                    mode='lines',
                    name=f'💼 {fund} ({optimal_weights[i]:.1%})',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.7,
                    hovertemplate=f'<b>{fund}</b><br>Weight: {optimal_weights[i]:.1%}<br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
                ))
        
        fig.update_layout(
            title="Cumulative Returns Comparison (%)",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary when individual funds are shown
        if show_individual_funds:
            st.subheader("📋 Individual Fund Performance Summary")
            
            fund_performance_summary = []
            for i, fund in enumerate(selected_funds):
                fund_returns_series = fund_returns[fund].fillna(0)
                fund_cumulative = (1 + fund_returns_series).cumprod()
                
                total_return = (fund_cumulative.iloc[-1] - 1) * 100 if len(fund_cumulative) > 0 else 0
                weight = optimal_weights[i]
                contribution = total_return * weight
                
                fund_performance_summary.append({
                    'Fund': fund,
                    'Weight': f"{weight:.1%}",
                    'Total Return': f"{total_return:.2f}%",
                    'Contribution to Portfolio': f"{contribution:.2f}%",
                    'Up Market Alpha': f"{metrics[fund]['up_alpha']:.2%}",
                    'Down Market Alpha': f"{metrics[fund]['down_alpha']:.2%}"
                })
            
            st.dataframe(pd.DataFrame(fund_performance_summary), use_container_width=True)
        
        # Raw data display section
        if show_raw_data:
            st.subheader("📊 Raw Calculation Data")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Fund Returns", "📊 Index Returns", "🔢 Price Data", "📋 Market Regime"])
            
            with tab1:
                st.write("**Daily Returns for Selected Funds (%)**")
                fund_returns_display = fund_returns[selected_funds] * 100  # Convert to percentage
                st.dataframe(fund_returns_display.round(4), use_container_width=True)
                
                st.write("**Fund Returns Statistics**")
                stats_df = fund_returns_display.describe()
                st.dataframe(stats_df, use_container_width=True)
            
            with tab2:
                st.write("**Index Daily Returns (%)**")
                index_returns_display = (index_returns * 100).round(4)
                st.dataframe(index_returns_display.to_frame('Index_Returns'), use_container_width=True)
                
                st.write("**Index Statistics**")
                index_stats = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                    'Value': [
                        f"{index_returns.mean()*100:.4f}%",
                        f"{index_returns.std()*100:.4f}%", 
                        f"{index_returns.min()*100:.4f}%",
                        f"{index_returns.max()*100:.4f}%",
                        f"{index_returns.skew():.4f}",
                        f"{index_returns.kurtosis():.4f}"
                    ]
                })
                st.dataframe(index_stats, use_container_width=True)
            
            with tab3:
                st.write("**Raw Price Data**")
                price_data = pd.DataFrame()
                price_data['Index_Price'] = index_data
                
                for fund in selected_funds:
                    if fund in fund_data:
                        price_data[f'{fund}_Price'] = fund_data[fund]
                
                st.dataframe(price_data.round(2), use_container_width=True)
            
            with tab4:
                st.write("**Market Regime Analysis**")
                regime_data = pd.DataFrame({
                    'Date': index_returns.index,
                    'Index_Return': index_returns.values,
                    'Market_Regime': ['Up' if ret > 0 else 'Down' for ret in index_returns.values],
                    'Cumulative_Return': (1 + index_returns).cumprod().values - 1
                })
                st.dataframe(regime_data.round(4), use_container_width=True)
        
        # Risk Analysis
        st.subheader("🛡️ Risk Analysis")
        
        portfolio_vol = portfolio_daily_returns.std() * np.sqrt(252)
        index_vol = index_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_daily_returns.mean() * 252) / portfolio_vol if portfolio_vol > 0 else 0
        index_sharpe = (index_returns.mean() * 252) / index_vol if index_vol > 0 else 0
        
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        
        with risk_col1:
            st.metric("Portfolio Volatility", f"{portfolio_vol:.1%}")
        
        with risk_col2:
            st.metric("Index Volatility", f"{index_vol:.1%}")
        
        with risk_col3:
            st.metric("Portfolio Sharpe", f"{portfolio_sharpe:.2f}")
        
        with risk_col4:
            st.metric("Index Sharpe", f"{index_sharpe:.2f}")
        
        # Detailed fund analysis
        with st.expander("🔍 Detailed Fund Analysis"):
            detailed_analysis = []
            for fund in selected_funds:
                m = metrics[fund]
                detailed_analysis.append({
                    'Fund': fund,
                    'Weight': f"{optimal_weights[selected_funds.index(fund)]:.1%}",
                    'Up Market Return': f"{m['up_performance']:.2%}",
                    'Down Market Return': f"{m['down_performance']:.2%}",
                    'Up Market Alpha': f"{m['up_alpha']:.2%}",
                    'Down Market Alpha': f"{m['down_alpha']:.2%}",
                    'Volatility': f"{m['volatility']:.2%}",
                    'Sharpe Ratio': f"{m['sharpe']:.2f}",
                    'Score': f"{m['score']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(detailed_analysis), use_container_width=True)

if __name__ == "__main__":
    main()
