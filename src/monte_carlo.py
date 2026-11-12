"""
Monte Carlo Simulation Engine: 10,000+ Iteration Portfolio Scenario Generator.
Simulates terminal distribution paths, stress events, and tail-risk exposure.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd


class MonteCarloEngine:
    """Vectorized Monte Carlo path generation for multi-asset portfolios."""

    def __init__(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        initial_investment: float = 1_000_000.0,
        seed: int = 42,
    ):
        self.mean_returns = np.asarray(mean_returns)
        self.cov_matrix = np.asarray(cov_matrix)
        self.weights = np.asarray(weights)
        self.initial_investment = initial_investment
        self.seed = seed

    def run_simulations(
        self,
        iterations: int = 10_000,
        time_horizon_days: int = 252,
    ) -> Dict[str, object]:
        """
        Execute vectorized Monte Carlo simulations across portfolio assets.
        Returns path trajectories, percentiles, terminal portfolio values, and tail statistics.
        """
        np.random.seed(self.seed)
        n_assets = len(self.weights)

        # Cholesky decomposition of the covariance matrix for correlated daily draws
        L = np.linalg.cholesky(self.cov_matrix)

        # Generate correlated standard normal random variables: shape (iterations, days, n_assets)
        random_normals = np.random.normal(size=(iterations, time_horizon_days, n_assets))
        correlated_daily_returns = np.einsum("ijk,lk->ijl", random_normals, L) + self.mean_returns

        # Multiply by weights to obtain daily portfolio return per simulation
        # shape: (iterations, time_horizon_days)
        portfolio_daily_returns = np.dot(correlated_daily_returns, self.weights)

        # Compound returns across the time horizon
        # Shape: (iterations, time_horizon_days + 1)
        cumulative_growth = np.cumprod(1.0 + portfolio_daily_returns, axis=1)
        trajectories = np.hstack([np.ones((iterations, 1)), cumulative_growth]) * self.initial_investment

        terminal_values = trajectories[:, -1]

        # Calculate percentile distributions
        p1 = np.percentile(terminal_values, 1.0)
        p5 = np.percentile(terminal_values, 5.0)
        p25 = np.percentile(terminal_values, 25.0)
        p50 = np.percentile(terminal_values, 50.0)
        p75 = np.percentile(terminal_values, 75.0)
        p95 = np.percentile(terminal_values, 95.0)
        p99 = np.percentile(terminal_values, 99.0)

        # Probability of capital impairment (< initial_investment)
        prob_of_loss = float(np.mean(terminal_values < self.initial_investment))

        # Expected Tail Loss (CVaR equivalent from terminal values)
        var_95_val = self.initial_investment - p5
        tail_95_losses = self.initial_investment - terminal_values[terminal_values <= p5]
        cvar_95_val = float(np.mean(tail_95_losses)) if len(tail_95_losses) > 0 else var_95_val

        return {
            "iterations": iterations,
            "horizon_days": time_horizon_days,
            "trajectories": trajectories,  # Shape (iterations, days+1)
            "terminal_values": terminal_values,
            "mean_terminal_value": float(np.mean(terminal_values)),
            "std_terminal_value": float(np.std(terminal_values)),
            "percentiles": {
                "p1": p1,
                "p5": p5,
                "p25": p25,
                "median_p50": p50,
                "p75": p75,
                "p95": p95,
                "p99": p99,
            },
            "prob_of_loss": prob_of_loss,
            "mc_var_95": max(var_95_val, 0.0),
            "mc_cvar_95": max(cvar_95_val, 0.0),
        }
