"""
Orchestration Pipeline: AI-Driven Quantitative Portfolio Risk Advisor.
Unifies Market Data Ingestion, Markowitz Optimization, NLP Sentiment Tilting,
10,000+ Monte Carlo Simulations, and Risk Reporting.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.data_loader import MarketDataLoader
from src.markowitz import MarkowitzOptimizer
from src.monte_carlo import MonteCarloEngine
from src.risk_metrics import RiskMetricsCalculator
from src.sentiment_alpha import SentimentAlphaEngine


class PortfolioRiskAdvisor:
    """End-to-end advisory engine for institutional quantitative risk management."""

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        portfolio_value: float = 1_000_000.0,
        risk_free_rate: float = 0.04,
        tilt_intensity: float = 0.25,
    ):
        self.tickers = tickers or ["AAPL", "MSFT", "GOOGL", "NVDA", "GLD", "BND"]
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate
        self.loader = MarketDataLoader(self.tickers)
        self.sentiment_engine = SentimentAlphaEngine(tilt_intensity=tilt_intensity)

    def run_advisory_pipeline(
        self,
        prices_df: Optional[pd.DataFrame] = None,
        ticker_headlines: Optional[Dict[str, List[str]]] = None,
        mc_iterations: int = 10_000,
        mc_horizon_days: int = 252,
    ) -> Dict[str, object]:
        """Executes full quantitative analysis and generates comprehensive recommendations."""
        # 1. Fetch or generate price history
        if prices_df is None:
            prices_df = self.loader.generate_synthetic_data()

        daily_returns = self.loader.compute_daily_returns(prices_df)

        # 2. Markowitz MPT Optimization
        optimizer = MarkowitzOptimizer(daily_returns, risk_free_rate=self.risk_free_rate)
        max_sharpe_res = optimizer.max_sharpe_portfolio()
        min_var_res = optimizer.min_variance_portfolio()
        efficient_frontier = optimizer.calculate_efficient_frontier(n_points=35)

        # 3. NLP Sentiment Alpha Signals
        sentiment_scores = {}
        if ticker_headlines:
            sentiment_scores = self.sentiment_engine.evaluate_ticker_sentiment(ticker_headlines)
        else:
            # Default realistic market sentiment signals
            sentiment_scores = {
                "AAPL": 0.45,   # Positive earnings beat
                "MSFT": 0.30,   # Cloud revenue growth
                "GOOGL": 0.10,  # Stable search
                "NVDA": 0.65,   # AI hardware demand surge
                "GLD": -0.15,   # Moderate risk-on sentiment
                "BND": -0.05,   # Yield stability
            }

        tilted_weights_dict = self.sentiment_engine.apply_sentiment_tilt(
            max_sharpe_res["weights"],
            sentiment_scores,
        )
        tilted_weights_array = np.array([tilted_weights_dict[t] for t in self.tickers])

        # 4. Risk Metrics for Base & Tilted Portfolios
        base_risk_report = RiskMetricsCalculator.compute_comprehensive_risk_report(
            daily_returns,
            max_sharpe_res["weights_array"],
            portfolio_value=self.portfolio_value,
            risk_free_rate=self.risk_free_rate,
        )

        tilted_risk_report = RiskMetricsCalculator.compute_comprehensive_risk_report(
            daily_returns,
            tilted_weights_array,
            portfolio_value=self.portfolio_value,
            risk_free_rate=self.risk_free_rate,
        )

        # 5. Monte Carlo 10,000+ Simulations on Recommended Allocation
        mc_engine = MonteCarloEngine(
            mean_returns=optimizer.mean_daily_returns,
            cov_matrix=optimizer.daily_cov_matrix,
            weights=tilted_weights_array,
            initial_investment=self.portfolio_value,
        )
        mc_results = mc_engine.run_simulations(
            iterations=mc_iterations,
            time_horizon_days=mc_horizon_days,
        )

        return {
            "tickers": self.tickers,
            "prices": prices_df,
            "returns": daily_returns,
            "max_sharpe_portfolio": max_sharpe_res,
            "min_variance_portfolio": min_var_res,
            "efficient_frontier": efficient_frontier,
            "sentiment_scores": sentiment_scores,
            "tilted_weights": tilted_weights_dict,
            "base_risk_report": base_risk_report,
            "tilted_risk_report": tilted_risk_report,
            "monte_carlo_results": mc_results,
        }
