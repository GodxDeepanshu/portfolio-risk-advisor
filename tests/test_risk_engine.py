"""
Unit & Integration Tests for AI-Driven Portfolio Risk Advisor.
Validates Markowitz Optimization, VaR/CVaR, Monte Carlo Simulation, and NLP Sentiment Tilting.
"""
import pytest
import numpy as np
import pandas as pd

from src.data_loader import MarketDataLoader
from src.markowitz import MarkowitzOptimizer
from src.monte_carlo import MonteCarloEngine
from src.risk_metrics import RiskMetricsCalculator
from src.sentiment_alpha import SentimentAlphaEngine
from src.portfolio_advisor import PortfolioRiskAdvisor


@pytest.fixture
def sample_market_data():
    loader = MarketDataLoader(tickers=["AAPL", "MSFT", "GOOGL"], seed=101)
    prices = loader.generate_synthetic_data(periods=252)
    returns = loader.compute_daily_returns(prices)
    return prices, returns


def test_market_data_generation(sample_market_data):
    prices, returns = sample_market_data
    assert prices.shape[0] == 252
    assert prices.shape[1] == 3
    assert not prices.isnull().values.any()
    assert returns.shape[0] == 251


def test_markowitz_optimization(sample_market_data):
    _, returns = sample_market_data
    optimizer = MarkowitzOptimizer(returns, risk_free_rate=0.04)
    res_sharpe = optimizer.max_sharpe_portfolio()
    res_min_var = optimizer.min_variance_portfolio()

    # Weights must sum to 1.0 (simplex constraint)
    assert np.isclose(sum(res_sharpe["weights"].values()), 1.0, atol=1e-3)
    assert np.isclose(sum(res_min_var["weights"].values()), 1.0, atol=1e-3)

    # Minimum variance portfolio volatility must be <= max sharpe portfolio volatility
    assert res_min_var["volatility"] <= res_sharpe["volatility"] + 1e-4

    # Efficient frontier returns valid points
    frontier = optimizer.calculate_efficient_frontier(n_points=10)
    assert len(frontier) > 0
    assert all("return" in p and "volatility" in p for p in frontier)


def test_risk_metrics_calculations(sample_market_data):
    _, returns = sample_market_data
    weights = np.array([0.4, 0.4, 0.2])
    report = RiskMetricsCalculator.compute_comprehensive_risk_report(
        returns,
        weights,
        portfolio_value=1_000_000.0,
        risk_free_rate=0.04,
    )

    # Assert metric validity
    assert report["parametric_var_95"] > 0
    assert report["parametric_var_99"] > report["parametric_var_95"]
    assert report["historical_cvar_95"] >= report["historical_var_95"]
    assert 0 <= report["maximum_drawdown"] <= 1.0
    assert isinstance(report["sharpe_ratio"], float)


def test_monte_carlo_engine(sample_market_data):
    _, returns = sample_market_data
    weights = np.array([0.5, 0.3, 0.2])
    mean_ret = returns.mean().values
    cov_mat = returns.cov().values

    mc = MonteCarloEngine(
        mean_returns=mean_ret,
        cov_matrix=cov_mat,
        weights=weights,
        initial_investment=1_000_000.0,
        seed=42,
    )
    results = mc.run_simulations(iterations=1000, time_horizon_days=50)

    assert results["trajectories"].shape == (1000, 51)
    assert len(results["terminal_values"]) == 1000
    assert results["percentiles"]["p95"] > results["percentiles"]["p5"]
    assert 0.0 <= results["prob_of_loss"] <= 1.0


def test_sentiment_alpha_engine():
    engine = SentimentAlphaEngine(tilt_intensity=0.20)
    # Test headline scoring
    bullish_score = engine.score_headline("Company reports record profit and massive revenue surge")
    bearish_score = engine.score_headline("Company faces severe loss, layoffs, and regulatory lawsuit")

    assert bullish_score > 0.3
    assert bearish_score < -0.3

    # Test tilt adjustment
    base_weights = {"AAPL": 0.5, "MSFT": 0.5}
    sentiment = {"AAPL": 0.8, "MSFT": -0.8}
    tilted = engine.apply_sentiment_tilt(base_weights, sentiment)

    assert tilted["AAPL"] > tilted["MSFT"]
    assert np.isclose(sum(tilted.values()), 1.0, atol=1e-3)


def test_end_to_end_advisor():
    advisor = PortfolioRiskAdvisor(tickers=["AAPL", "MSFT", "GOOGL"])
    res = advisor.run_advisory_pipeline(mc_iterations=500, mc_horizon_days=30)
    assert "max_sharpe_portfolio" in res
    assert "monte_carlo_results" in res
    assert "tilted_risk_report" in res
