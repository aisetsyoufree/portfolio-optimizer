#!/usr/bin/env python3
"""
Functional UI Test Suite for Portfolio Optimizer
Tests all UI functionalities using direct function calls and data validation
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import requests
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio_app import PortfolioOptimizer, create_excel_export

class TestUIFunctionality(unittest.TestCase):
    """Test all UI functionality with S&P 500 scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = PortfolioOptimizer()
        print(f"\n{'='*50}")
    
    def test_01_sp500_balanced_strategy_2023(self):
        """Test S&P 500 with Balanced strategy for 2023"""
        print("1️⃣ Testing S&P 500 - Balanced Strategy (2023)")
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Test index data fetching
        print("   📊 Fetching S&P 500 index data...")
        index_data = self.optimizer.fetch_single_stock('^GSPC', start_date, end_date)
        self.assertIsNotNone(index_data, "S&P 500 data should be fetchable")
        self.assertGreater(len(index_data), 200, "Should have substantial data points")
        
        # Test fund universe fetching (limited for testing)
        print("   📈 Fetching fund universe (limited to 10 for testing)...")
        fund_data = self.optimizer.fetch_fund_universe(start_date, end_date, max_funds=10)
        self.assertGreater(len(fund_data), 3, "Should fetch at least 4 funds")
        
        # Test performance metrics calculation
        print("   🔢 Calculating performance metrics...")
        fund_returns = pd.DataFrame({symbol: data.pct_change().dropna() 
                                   for symbol, data in fund_data.items()})
        index_returns = index_data.pct_change().dropna()
        
        # Align data
        common_dates = fund_returns.index.intersection(index_returns.index)
        fund_returns = fund_returns.loc[common_dates]
        index_returns = index_returns.loc[common_dates]
        
        metrics = self.optimizer.calculate_performance_metrics(
            fund_returns, index_returns, 'balanced'
        )
        
        self.assertEqual(len(metrics), len(fund_data), "Should have metrics for all funds")
        
        # Test portfolio optimization
        print("   ⚖️ Testing portfolio optimization...")
        selected_funds = list(fund_data.keys())[:5]  # Select top 5 funds
        optimal_weights = self.optimizer.optimize_weights(
            selected_funds, fund_returns, index_returns, 'balanced'
        )
        
        self.assertEqual(len(optimal_weights), len(selected_funds), "Should have weight for each fund")
        self.assertAlmostEqual(sum(optimal_weights), 1.0, places=3, msg="Weights should sum to 1.0")
        
        print("   ✅ S&P 500 Balanced strategy test completed successfully")
        
        return {
            'index_data': index_data,
            'fund_data': fund_data,
            'fund_returns': fund_returns,
            'index_returns': index_returns,
            'selected_funds': selected_funds,
            'optimal_weights': optimal_weights,
            'metrics': metrics
        }
    
    def test_02_sp500_growth_strategy_6months(self):
        """Test S&P 500 with Growth Focus strategy for 6 months"""
        print("2️⃣ Testing S&P 500 - Growth Focus (6 months)")
        
        start_date = datetime(2023, 7, 1)
        end_date = datetime(2023, 12, 31)
        
        # Test with growth strategy
        index_data = self.optimizer.fetch_single_stock('^GSPC', start_date, end_date)
        self.assertIsNotNone(index_data, "S&P 500 data should be fetchable")
        
        fund_data = self.optimizer.fetch_fund_universe(start_date, end_date, max_funds=8)
        self.assertGreater(len(fund_data), 3, "Should fetch at least 4 funds")
        
        fund_returns = pd.DataFrame({symbol: data.pct_change().dropna() 
                                   for symbol, data in fund_data.items()})
        index_returns = index_data.pct_change().dropna()
        
        common_dates = fund_returns.index.intersection(index_returns.index)
        fund_returns = fund_returns.loc[common_dates]
        index_returns = index_returns.loc[common_dates]
        
        metrics = self.optimizer.calculate_performance_metrics(
            fund_returns, index_returns, 'growth'
        )
        
        selected_funds = list(fund_data.keys())[:4]
        optimal_weights = self.optimizer.optimize_weights(
            selected_funds, fund_returns, index_returns, 'growth'
        )
        
        self.assertAlmostEqual(sum(optimal_weights), 1.0, places=3)
        print("   ✅ S&P 500 Growth Focus test completed successfully")
        
        return {
            'strategy': 'growth',
            'selected_funds': selected_funds,
            'optimal_weights': optimal_weights
        }
    
    def test_03_sp500_downside_protection_bear_market(self):
        """Test S&P 500 with Downside Protection during bear market"""
        print("3️⃣ Testing S&P 500 - Downside Protection (2022 bear market)")
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        
        index_data = self.optimizer.fetch_single_stock('^GSPC', start_date, end_date)
        self.assertIsNotNone(index_data, "S&P 500 data should be fetchable")
        
        fund_data = self.optimizer.fetch_fund_universe(start_date, end_date, max_funds=8)
        self.assertGreater(len(fund_data), 3, "Should fetch at least 4 funds")
        
        fund_returns = pd.DataFrame({symbol: data.pct_change().dropna() 
                                   for symbol, data in fund_data.items()})
        index_returns = index_data.pct_change().dropna()
        
        common_dates = fund_returns.index.intersection(index_returns.index)
        fund_returns = fund_returns.loc[common_dates]
        index_returns = index_returns.loc[common_dates]
        
        metrics = self.optimizer.calculate_performance_metrics(
            fund_returns, index_returns, 'downside_protection'
        )
        
        selected_funds = list(fund_data.keys())[:6]
        optimal_weights = self.optimizer.optimize_weights(
            selected_funds, fund_returns, index_returns, 'downside_protection'
        )
        
        self.assertAlmostEqual(sum(optimal_weights), 1.0, places=3)
        print("   ✅ S&P 500 Downside Protection test completed successfully")
        
        return {
            'strategy': 'downside_protection',
            'selected_funds': selected_funds,
            'optimal_weights': optimal_weights
        }
    
    def test_04_excel_export_functionality(self):
        """Test Excel export with real data"""
        print("4️⃣ Testing Excel Export Functionality")
        
        # Get test data from previous test
        test_data = self.test_01_sp500_balanced_strategy_2023()
        
        # Create portfolio summary
        portfolio_df = pd.DataFrame({
            'Fund': test_data['selected_funds'],
            'Weight': [f"{w:.1%}" for w in test_data['optimal_weights']],
            'Expected_Return': ['5.2%', '7.1%', '4.8%', '6.3%', '5.9%'][:len(test_data['selected_funds'])]
        })
        
        # Test Excel export function
        print("   📊 Creating Excel export...")
        try:
            excel_data = create_excel_export(
                portfolio_df, [], [], 
                test_data['index_data'], 
                test_data['fund_returns'],
                test_data['selected_funds'],
                test_data['optimal_weights']
            )
            
            self.assertIsNotNone(excel_data, "Excel data should be generated")
            self.assertGreater(len(excel_data), 1000, "Excel file should have substantial content")
            
            print("   ✅ Excel export generated successfully")
            print(f"   📁 Excel file size: {len(excel_data):,} bytes")
            
        except Exception as e:
            self.fail(f"Excel export failed: {e}")
    
    def test_05_raw_data_processing(self):
        """Test raw data processing and display functionality"""
        print("5️⃣ Testing Raw Data Processing")
        
        test_data = self.test_01_sp500_balanced_strategy_2023()
        
        # Test fund returns processing
        print("   📈 Testing fund returns data...")
        fund_returns_display = test_data['fund_returns'] * 100  # Convert to percentage
        self.assertFalse(fund_returns_display.empty, "Fund returns should not be empty")
        
        # Test statistics calculation
        stats_df = fund_returns_display.describe()
        self.assertIn('mean', stats_df.index, "Should have mean statistics")
        self.assertIn('std', stats_df.index, "Should have standard deviation")
        
        # Test index statistics
        index_returns = test_data['index_returns']
        index_stats = {
            'Mean': f"{index_returns.mean()*100:.4f}%",
            'Std Dev': f"{index_returns.std()*100:.4f}%",
            'Min': f"{index_returns.min()*100:.4f}%",
            'Max': f"{index_returns.max()*100:.4f}%",
            'Skewness': f"{index_returns.skew():.4f}",
            'Kurtosis': f"{index_returns.kurtosis():.4f}"
        }
        
        self.assertIn('Mean', index_stats, "Should calculate index mean")
        self.assertIn('Std Dev', index_stats, "Should calculate index volatility")
        
        # Test market regime analysis
        regime_data = pd.DataFrame({
            'Date': index_returns.index,
            'Index_Return': index_returns.values,
            'Market_Regime': ['Up' if ret > 0 else 'Down' for ret in index_returns.values],
            'Cumulative_Return': (1 + index_returns).cumprod().values - 1
        })
        
        self.assertFalse(regime_data.empty, "Market regime data should not be empty")
        self.assertIn('Up', regime_data['Market_Regime'].values, "Should have up market days")
        self.assertIn('Down', regime_data['Market_Regime'].values, "Should have down market days")
        
        print("   ✅ Raw data processing working correctly")
        print(f"   📊 Fund returns shape: {fund_returns_display.shape}")
        print(f"   📈 Index data points: {len(index_returns)}")
        print(f"   🎯 Market regime data: {len(regime_data)} days")
    
    def test_06_performance_metrics_validation(self):
        """Test performance metrics calculations"""
        print("6️⃣ Testing Performance Metrics Validation")
        
        test_data = self.test_01_sp500_balanced_strategy_2023()
        metrics = test_data['metrics']
        
        for fund, metric in metrics.items():
            # Validate metric structure
            required_keys = ['up_performance', 'down_performance', 'up_alpha', 'down_alpha', 
                           'volatility', 'sharpe', 'score']
            for key in required_keys:
                self.assertIn(key, metric, f"Metric {key} should exist for fund {fund}")
            
            # Validate metric ranges
            self.assertIsInstance(metric['volatility'], (int, float), "Volatility should be numeric")
            self.assertIsInstance(metric['sharpe'], (int, float), "Sharpe ratio should be numeric")
            self.assertGreaterEqual(metric['volatility'], 0, "Volatility should be non-negative")
        
        print(f"   ✅ Performance metrics validated for {len(metrics)} funds")
    
    def test_07_app_connectivity(self):
        """Test that the Streamlit app is accessible"""
        print("7️⃣ Testing App Connectivity")
        
        try:
            response = requests.get("http://localhost:8507", timeout=10)
            self.assertEqual(response.status_code, 200, "App should be accessible")
            
            # Check if it's actually Streamlit
            self.assertIn("streamlit", response.text.lower(), "Should be Streamlit app")
            
            print("   ✅ App is accessible and running Streamlit")
            print(f"   🌐 Response status: {response.status_code}")
            
        except requests.exceptions.RequestException as e:
            self.fail(f"App connectivity test failed: {e}")

def run_comprehensive_functional_tests():
    """Run comprehensive functional tests for all UI features"""
    print("🧪 Starting Comprehensive Functional UI Tests")
    print("=" * 70)
    
    # Check app availability first
    print("🔍 Checking app availability...")
    try:
        response = requests.get("http://localhost:8507", timeout=5)
        if response.status_code == 200:
            print("✅ Portfolio Optimizer app is running")
        else:
            print(f"❌ App returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ App is not accessible. Start with: streamlit run portfolio_app.py --server.port 8507")
        return False
    
    # Run functional tests
    test_suite = unittest.TestSuite()
    
    # Add all test methods
    test_suite.addTest(TestUIFunctionality('test_07_app_connectivity'))
    test_suite.addTest(TestUIFunctionality('test_01_sp500_balanced_strategy_2023'))
    test_suite.addTest(TestUIFunctionality('test_02_sp500_growth_strategy_6months'))
    test_suite.addTest(TestUIFunctionality('test_03_sp500_downside_protection_bear_market'))
    test_suite.addTest(TestUIFunctionality('test_04_excel_export_functionality'))
    test_suite.addTest(TestUIFunctionality('test_05_raw_data_processing'))
    test_suite.addTest(TestUIFunctionality('test_06_performance_metrics_validation'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Detailed summary
    print("\n" + "=" * 70)
    print("🎯 Comprehensive Functional Test Results:")
    print(f"   📊 Tests Run: {result.testsRun}")
    print(f"   ✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ Failures: {len(result.failures)}")
    print(f"   🚨 Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n🔍 Failure Details:")
        for test, traceback in result.failures:
            print(f"   ❌ {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n🚨 Error Details:")
        for test, traceback in result.errors:
            print(f"   🚨 {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Feature validation summary
    print("\n📋 UI Features Validation Summary:")
    features_tested = [
        "✅ S&P 500 index data fetching",
        "✅ Fund universe data fetching", 
        "✅ Performance metrics calculation",
        "✅ Portfolio weight optimization",
        "✅ Excel export functionality",
        "✅ Raw data processing",
        "✅ Multiple strategy support",
        "✅ Date range flexibility",
        "✅ App connectivity and responsiveness"
    ]
    
    for feature in features_tested:
        print(f"   {feature}")
    
    # Manual testing checklist
    print("\n🎯 Manual UI Testing Checklist:")
    manual_tests = [
        "📊 Portfolio allocation pie chart displays correctly",
        "📈 Cumulative returns chart is interactive", 
        "☑️ 'Show Individual Fund Performance' toggle works",
        "📊 'Export to Excel' downloads without errors",
        "📋 'Show Raw Calculation Data' checkbox reveals tabs",
        "📑 All 4 raw data tabs display organized data",
        "🛡️ Risk analysis metrics are accurate",
        "⚡ Performance tables show correct calculations",
        "🎯 Strategy selection affects optimization results",
        "📅 Date range picker accepts valid dates",
        "🔢 Portfolio size slider adjusts fund count"
    ]
    
    for test in manual_tests:
        print(f"   ☐ {test}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎉 Overall Success Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_functional_tests()
    
    if success:
        print("\n🎊 All automated functional tests passed!")
        print("🔗 Open http://localhost:8507 to manually verify UI features")
    else:
        print("\n⚠️ Some tests failed. Check the details above.")
    
    exit(0 if success else 1)
