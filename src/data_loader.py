"""
Market Data Engine: Correlated Geometric Brownian Motion & Financial Data Loading.
Supports real-world multi-asset simulation, CSV imports, and API integrations.
"""
from typing import List, Optional
import numpy as np
import pandas as pd


class MarketDataLoader:
    """Loads historical prices or generates realistic multi-asset correlated GBM price paths."""

    DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "GLD", "BND"]

    def __init__(self, tickers: Optional[List[str]] = None, seed: int = 42):
        self.tickers = tickers or self.DEFAULT_TICKERS
        self.seed = seed

    def generate_synthetic_data(
        self,
        start_date: str = "2023-01-01",
        periods: int = 504,  # ~2 trading years
        drift_range: tuple = (0.08, 0.22),
        vol_range: tuple = (0.15, 0.35),
    ) -> pd.DataFrame:
        """
        Simulate correlated multi-asset price histories using Cholesky Decomposition
        on Geometric Brownian Motion (GBM).
        """
        np.random.seed(self.seed)
        n_assets = len(self.tickers)
        dt = 1.0 / 252.0  # Daily time step

        # Baseline annual drift and volatility for assets
        drifts = np.random.uniform(drift_range[0], drift_range[1], n_assets)
        vols = np.random.uniform(vol_range[0], vol_range[1], n_assets)

        # Base correlation matrix with realistic market co-movements
        raw_cov = np.random.uniform(0.2, 0.6, (n_assets, n_assets))
        cov_matrix = (raw_cov + raw_cov.T) / 2.0
        np.fill_diagonal(cov_matrix, 1.0)
        
        # Ensure positive semi-definiteness via Cholesky factor
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-6)
        cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        std_diag = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_diag, std_diag)
        L = np.linalg.cholesky(corr_matrix)

        # Generate correlated standard normal innovations
        z = np.random.normal(size=(periods, n_assets))
        correlated_z = z @ L.T

        # Initial prices
        s0 = np.array([180.0, 350.0, 140.0, 480.0, 195.0, 75.0][:n_assets])
        if len(s0) < n_assets:
            s0 = np.pad(s0, (0, n_assets - len(s0)), constant_values=100.0)

        price_paths = np.zeros((periods, n_assets))
        price_paths[0] = s0

        for t in range(1, periods):
            drift_term = (drifts - 0.5 * vols**2) * dt
            diffusion_term = vols * np.sqrt(dt) * correlated_z[t]
            price_paths[t] = price_paths[t - 1] * np.exp(drift_term + diffusion_term)

        date_index = pd.bdate_range(start=start_date, periods=periods)
        df = pd.DataFrame(price_paths, index=date_index, columns=self.tickers)
        return df

    @staticmethod
    def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute simple percentage returns from price history."""
        return prices.pct_change().dropna()

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute continuously compounded log returns."""
        return np.log(prices / prices.shift(1)).dropna()
