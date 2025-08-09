# 🚀 Smart Portfolio Optimizer

A sophisticated portfolio optimization application that creates optimal fund portfolios designed to outperform market indices in both up and down market conditions.

## Features

- **Market Index Analysis**: Select from major indices (S&P 500, NASDAQ, Dow Jones, Russell 2000, VIX)
- **Date-Specific Optimization**: Choose any historical date for analysis
- **Comprehensive Fund Universe**: Analyzes 40+ popular ETFs and 16+ mutual funds
- **Dual-Regime Optimization**: Optimizes for outperformance in both bull and bear markets
- **Risk Analysis**: Provides volatility, Sharpe ratio, and other risk metrics
- **Interactive Visualizations**: Beautiful charts showing portfolio allocation and performance
- **Real-Time Data**: Uses Yahoo Finance API for up-to-date market data

## How It Works

1. **Input Selection**: Choose a market index and target date
2. **Data Fetching**: Downloads historical returns for the index and all available funds
3. **Market Regime Detection**: Determines if the market was in an up or down trend around your selected date
4. **Portfolio Optimization**: Uses advanced optimization algorithms to find the best 3-5 fund portfolio that:
   - Maximizes outperformance vs. the selected index
   - Performs well in both up and down markets
   - Maintains proper diversification
5. **Results Display**: Shows optimal weights, performance metrics, and risk analysis

## Installation

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the displayed URL (typically `http://localhost:8501`)

3. Use the sidebar to:
   - Select your benchmark index
   - Choose a target date for analysis
   - Set the number of funds (3-5) for your portfolio

4. Click "🔍 Analyze & Optimize" to generate your optimized portfolio

## Fund Universe

### ETFs Analyzed
- **Broad Market**: SPY, QQQ, IWM, VTI, ITOT
- **International**: VEA, VWO, EFA, EEM, IEFA, IEMG
- **Fixed Income**: BND, TLT, AGG, LQD, HYG, VTEB, VMOT
- **Sectors**: XLF, XLK, XLE, XLV, XLI, XLP, XLY, XLU, XLB, XLRE
- **Factors**: VIG, VYM, VTV, VUG
- **Size**: VB, VO, VXF, IJH, IJR
- **Commodities**: GLD, VNQ, VGSLX

### Mutual Funds Analyzed
- **Vanguard**: VTSAX, VTIAX, VBTLX, VGTSX, VTSMX, VFWAX, VTABX
- **Fidelity**: FXNAX, FSKAX, FTIHX
- **Schwab**: SWPPX, SWTSX, SWAGX, SWISX

## Key Metrics

- **Up Market Performance**: How the portfolio performs when the index is rising
- **Down Market Performance**: How the portfolio performs when the index is falling
- **Volatility**: Annual standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Cumulative Returns**: Long-term performance comparison

## Technical Details

- **Optimization Algorithm**: Sequential Least Squares Programming (SLSQP)
- **Lookback Period**: 1 year of historical data for optimization
- **Market Regime Detection**: 30-day window around target date
- **Diversification Constraint**: Maximum 40% allocation to any single fund
- **Data Source**: Yahoo Finance via yfinance library

## Disclaimer

This application is for educational and research purposes only. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## Requirements

- Python 3.8+
- Internet connection for real-time data fetching
- Modern web browser for Streamlit interface

## Support

For issues or questions, please check the code comments or create an issue in the repository.
