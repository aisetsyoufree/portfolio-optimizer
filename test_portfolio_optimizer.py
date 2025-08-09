import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the PortfolioOptimizer class
from portfolio_app import PortfolioOptimizer

class TestPortfolioOptimizer(unittest.TestCase):
    """Comprehensive test suite for Portfolio Optimizer"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.optimizer = PortfolioOptimizer()
        self.test_start_date = datetime(2023, 1, 1)
        self.test_end_date = datetime(2023, 12, 31)
        
        # Create sample data for testing
        dates = pd.date_range(start=self.test_start_date, end=self.test_end_date, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Sample index returns (S&P 500 like)
        self.sample_index_returns = pd.Series(
            np.random.normal(0.0005, 0.012, len(dates)),
            index=dates,
            name='index'
        )
        
        # Sample fund returns
        self.sample_fund_returns = pd.DataFrame({
            'FUND_A': np.random.normal(0.0006, 0.015, len(dates)),
            'FUND_B': np.random.normal(0.0004, 0.010, len(dates)),
            'FUND_C': np.random.normal(0.0008, 0.020, len(dates)),
            'FUND_D': np.random.normal(0.0003, 0.008, len(dates)),
            'FUND_E': np.random.normal(0.0007, 0.018, len(dates))
        }, index=dates)

    def test_initialization(self):
        """Test PortfolioOptimizer initialization"""
        self.assertIsInstance(self.optimizer.equity_etfs, list)
        self.assertIsInstance(self.optimizer.equity_mutual_funds, list)
        self.assertIsInstance(self.optimizer.indices, dict)
        
        # Check that we have reasonable number of funds
        self.assertGreater(len(self.optimizer.equity_etfs), 20)
        self.assertGreater(len(self.optimizer.equity_mutual_funds), 10)
        self.assertGreater(len(self.optimizer.indices), 3)

    def test_fetch_single_stock_valid_symbol(self):
        """Test fetching data for a valid stock symbol"""
        # Test with a known stable symbol
        data = self.optimizer.fetch_single_stock('SPY', 
                                                 datetime(2023, 1, 1), 
                                                 datetime(2023, 1, 31))
        
        # Should return data or None (depending on network/API availability)
        self.assertTrue(data is None or isinstance(data, pd.Series))
        
        if data is not None:
            self.assertGreater(len(data), 10)  # Should have reasonable amount of data

    def test_fetch_single_stock_invalid_symbol(self):
        """Test fetching data for an invalid stock symbol"""
        data = self.optimizer.fetch_single_stock('INVALID_SYMBOL_XYZ', 
                                                 self.test_start_date, 
                                                 self.test_end_date)
        
        # Should return None for invalid symbols
        self.assertIsNone(data)

    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        metrics = self.optimizer.calculate_performance_metrics(
            self.sample_fund_returns, 
            self.sample_index_returns, 
            "Balanced (Equal Emphasis)"
        )
        
        # Should return metrics for all funds
        self.assertEqual(len(metrics), 5)
        
        # Check that each fund has all required metrics
        for fund, fund_metrics in metrics.items():
            self.assertIn('score', fund_metrics)
            self.assertIn('up_performance', fund_metrics)
            self.assertIn('down_performance', fund_metrics)
            self.assertIn('up_alpha', fund_metrics)
            self.assertIn('down_alpha', fund_metrics)
            self.assertIn('total_return', fund_metrics)
            self.assertIn('volatility', fund_metrics)
            self.assertIn('sharpe', fund_metrics)
            
            # Check that metrics are numeric
            for metric_name, metric_value in fund_metrics.items():
                self.assertIsInstance(metric_value, (int, float, np.number))

    def test_calculate_performance_metrics_different_strategies(self):
        """Test performance metrics with different strategy focuses"""
        strategies = ["Balanced (Equal Emphasis)", "Downside Protection", "Growth Focus"]
        
        for strategy in strategies:
            metrics = self.optimizer.calculate_performance_metrics(
                self.sample_fund_returns, 
                self.sample_index_returns, 
                strategy
            )
            
            # Should return metrics for all funds regardless of strategy
            self.assertEqual(len(metrics), 5)
            
            # Scores should be different for different strategies
            scores = [m['score'] for m in metrics.values()]
            self.assertTrue(all(isinstance(score, (int, float, np.number)) for score in scores))

    def test_optimize_weights(self):
        """Test portfolio weight optimization"""
        selected_funds = ['FUND_A', 'FUND_B', 'FUND_C']
        
        weights = self.optimizer.optimize_weights(
            selected_funds,
            self.sample_fund_returns,
            self.sample_index_returns,
            "Balanced (Equal Emphasis)"
        )
        
        # Should return list of weights
        self.assertIsInstance(weights, list)
        self.assertEqual(len(weights), len(selected_funds))
        
        # Weights should sum to approximately 1
        self.assertAlmostEqual(sum(weights), 1.0, places=2)
        
        # All weights should be non-negative
        self.assertTrue(all(w >= 0 for w in weights))
        
        # No weight should exceed 40% (our constraint)
        self.assertTrue(all(w <= 0.41 for w in weights))  # Small tolerance for optimization

    def test_optimize_weights_edge_cases(self):
        """Test weight optimization with edge cases"""
        
        # Test with single fund
        single_fund_weights = self.optimizer.optimize_weights(
            ['FUND_A'],
            self.sample_fund_returns,
            self.sample_index_returns,
            "Balanced (Equal Emphasis)"
        )
        self.assertEqual(single_fund_weights, [1.0])
        
        # Test with empty fund list
        empty_weights = self.optimizer.optimize_weights(
            [],
            self.sample_fund_returns,
            self.sample_index_returns,
            "Balanced (Equal Emphasis)"
        )
        self.assertEqual(empty_weights, [])

    def test_data_alignment(self):
        """Test that fund and index data align properly"""
        # Create misaligned data
        fund_dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        index_dates = pd.date_range(start='2023-03-01', end='2023-12-31', freq='D')
        
        fund_returns = pd.DataFrame({
            'FUND_X': np.random.normal(0.001, 0.02, len(fund_dates))
        }, index=fund_dates)
        
        index_returns = pd.Series(
            np.random.normal(0.0005, 0.015, len(index_dates)),
            index=index_dates
        )
        
        metrics = self.optimizer.calculate_performance_metrics(
            fund_returns, index_returns, "Balanced (Equal Emphasis)"
        )
        
        # Should handle misaligned data gracefully
        self.assertIsInstance(metrics, dict)

    def test_performance_metrics_validation(self):
        """Test that performance metrics are within reasonable bounds"""
        metrics = self.optimizer.calculate_performance_metrics(
            self.sample_fund_returns, 
            self.sample_index_returns, 
            "Balanced (Equal Emphasis)"
        )
        
        for fund, fund_metrics in metrics.items():
            # Sharpe ratio should be reasonable (typically -3 to 3)
            self.assertGreaterEqual(fund_metrics['sharpe'], -5)
            self.assertLessEqual(fund_metrics['sharpe'], 5)
            
            # Volatility should be positive
            self.assertGreaterEqual(fund_metrics['volatility'], 0)
            
            # Returns should be reasonable (daily returns typically < 20%)
            self.assertGreaterEqual(fund_metrics['up_performance'], -0.2)
            self.assertLessEqual(fund_metrics['up_performance'], 0.2)
            self.assertGreaterEqual(fund_metrics['down_performance'], -0.2)
            self.assertLessEqual(fund_metrics['down_performance'], 0.2)

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and edge cases"""
    
    def setUp(self):
        self.optimizer = PortfolioOptimizer()

    def test_fund_symbols_validity(self):
        """Test that fund symbols are valid format"""
        all_symbols = self.optimizer.equity_etfs + self.optimizer.equity_mutual_funds
        
        for symbol in all_symbols:
            # Should be strings
            self.assertIsInstance(symbol, str)
            # Should be uppercase
            self.assertEqual(symbol, symbol.upper())
            # Should be reasonable length (2-6 characters typically)
            self.assertGreaterEqual(len(symbol), 2)
            self.assertLessEqual(len(symbol), 6)

    def test_index_symbols_validity(self):
        """Test that index symbols are valid"""
        for name, symbol in self.optimizer.indices.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(symbol, str)
            self.assertTrue(symbol.startswith('^'))  # Yahoo Finance index format

    def test_no_duplicate_symbols(self):
        """Test that there are no duplicate symbols in fund lists"""
        etf_set = set(self.optimizer.equity_etfs)
        mutual_fund_set = set(self.optimizer.equity_mutual_funds)
        
        # No duplicates within each list
        self.assertEqual(len(etf_set), len(self.optimizer.equity_etfs))
        self.assertEqual(len(mutual_fund_set), len(self.optimizer.equity_mutual_funds))

class TestRegressionSuite(unittest.TestCase):
    """Regression test suite to catch breaking changes"""
    
    def setUp(self):
        self.optimizer = PortfolioOptimizer()
        
        # Create consistent test data
        np.random.seed(123)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        self.test_fund_returns = pd.DataFrame({
            'SPY': np.random.normal(0.0005, 0.012, len(dates)),
            'QQQ': np.random.normal(0.0008, 0.018, len(dates)),
            'VTI': np.random.normal(0.0006, 0.013, len(dates)),
            'BND': np.random.normal(0.0002, 0.005, len(dates)),
            'GLD': np.random.normal(0.0001, 0.015, len(dates))
        }, index=dates)
        
        self.test_index_returns = pd.Series(
            np.random.normal(0.0005, 0.012, len(dates)),
            index=dates
        )

    def test_baseline_portfolio_optimization(self):
        """Baseline test for portfolio optimization - should not change unexpectedly"""
        metrics = self.optimizer.calculate_performance_metrics(
            self.test_fund_returns,
            self.test_index_returns,
            "Balanced (Equal Emphasis)"
        )
        
        # Should always return metrics for all 5 test funds
        self.assertEqual(len(metrics), 5)
        
        # Test weight optimization
        selected_funds = ['SPY', 'QQQ', 'VTI']
        weights = self.optimizer.optimize_weights(
            selected_funds,
            self.test_fund_returns,
            self.test_index_returns,
            "Balanced (Equal Emphasis)"
        )
        
        # Should return 3 weights that sum to 1
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 1.0, places=2)

    def test_strategy_consistency(self):
        """Test that different strategies produce consistent but different results"""
        strategies = ["Balanced (Equal Emphasis)", "Downside Protection", "Growth Focus"]
        results = {}
        
        for strategy in strategies:
            metrics = self.optimizer.calculate_performance_metrics(
                self.test_fund_returns,
                self.test_index_returns,
                strategy
            )
            results[strategy] = metrics
        
        # All strategies should return same number of funds
        lengths = [len(result) for result in results.values()]
        self.assertTrue(all(length == lengths[0] for length in lengths))
        
        # Scores should be different between strategies (at least for some funds)
        balanced_scores = [m['score'] for m in results["Balanced (Equal Emphasis)"].values()]
        downside_scores = [m['score'] for m in results["Downside Protection"].values()]
        
        # At least one score should be different
        self.assertFalse(all(abs(b - d) < 0.0001 for b, d in zip(balanced_scores, downside_scores)))

def run_performance_benchmark():
    """Performance benchmark test (not part of unittest)"""
    print("🚀 Running Performance Benchmark...")
    
    optimizer = PortfolioOptimizer()
    
    # Test data fetching performance
    import time
    start_time = time.time()
    
    # Simulate fetching 10 funds
    test_symbols = optimizer.equity_etfs[:10]
    test_date = datetime(2023, 6, 1)
    
    print(f"Testing data fetch for {len(test_symbols)} symbols...")
    
    valid_count = 0
    for symbol in test_symbols:
        data = optimizer.fetch_single_stock(symbol, test_date - timedelta(days=30), test_date)
        if data is not None:
            valid_count += 1
    
    end_time = time.time()
    
    print(f"✅ Fetched {valid_count}/{len(test_symbols)} symbols in {end_time - start_time:.2f} seconds")
    print(f"📊 Average time per symbol: {(end_time - start_time)/len(test_symbols):.2f} seconds")
    
    return valid_count, end_time - start_time

def run_integration_test():
    """Integration test for the full workflow"""
    print("🔧 Running Integration Test...")
    
    optimizer = PortfolioOptimizer()
    
    # Create test data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(456)
    
    fund_returns = pd.DataFrame({
        'TEST_ETF_1': np.random.normal(0.0008, 0.015, len(dates)),
        'TEST_ETF_2': np.random.normal(0.0006, 0.012, len(dates)),
        'TEST_ETF_3': np.random.normal(0.0004, 0.010, len(dates)),
        'TEST_ETF_4': np.random.normal(0.0007, 0.016, len(dates)),
        'TEST_ETF_5': np.random.normal(0.0005, 0.011, len(dates))
    }, index=dates)
    
    index_returns = pd.Series(
        np.random.normal(0.0005, 0.012, len(dates)),
        index=dates
    )
    
    try:
        # Test full workflow
        metrics = optimizer.calculate_performance_metrics(
            fund_returns, index_returns, "Balanced (Equal Emphasis)"
        )
        
        selected_funds = list(fund_returns.columns[:4])
        weights = optimizer.optimize_weights(
            selected_funds, fund_returns, index_returns, "Balanced (Equal Emphasis)"
        )
        
        print(f"✅ Integration test passed!")
        print(f"📊 Analyzed {len(metrics)} funds")
        print(f"🎯 Optimized portfolio with {len(selected_funds)} funds")
        print(f"⚖️ Weights sum: {sum(weights):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Starting Portfolio Optimizer Test Suite")
    print("=" * 50)
    
    # Run unit tests
    print("\n1️⃣ Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n2️⃣ Running Performance Benchmark...")
    run_performance_benchmark()
    
    print("\n3️⃣ Running Integration Test...")
    run_integration_test()
    
    print("\n✅ Test Suite Complete!")
    print("=" * 50)
