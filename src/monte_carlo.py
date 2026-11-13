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
        # A weighted sum of jointly normal returns is itself normal.
        mean = self.weights @ self.mean_returns
        variance = self.weights @ self.cov_matrix @ self.weights
        trajectories = np.empty((iterations, time_horizon_days + 1))
        trajectories[:, 0] = self.initial_investment
        growth = trajectories[:, 1:]
        rng = np.random.default_rng(self.seed)
        # Bound temporary memory while preserving the seeded draw sequence.
        for start in range(0, iterations, 1024):
            batch = growth[start:start + 1024]
            batch[:] = rng.normal(mean, np.sqrt(max(variance, 0.0)), size=batch.shape)
        np.maximum(growth, -1.0, out=growth)
        growth += 1.0
        np.cumprod(growth, axis=1, out=growth)
        growth *= self.initial_investment

        terminal_values = trajectories[:, -1]
        p1, p5, p25, p50, p75, p95, p99 = np.percentile(
            terminal_values, [1, 5, 25, 50, 75, 95, 99]
        )

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
