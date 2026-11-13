"""
Markowitz Modern Portfolio Theory & Efficient Frontier Optimization Engine.
Computes Maximum Sharpe Portfolio, Global Minimum Variance (GMV) Portfolio,
Efficient Frontier curve, and dynamic covariance shrinkage matrices.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MarkowitzOptimizer:
    """Solves continuous portfolio optimization problems subject to constraints."""

    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, returns_df: pd.DataFrame, risk_free_rate: float = 0.04):
        self.returns_df = returns_df
        self.tickers = list(returns_df.columns)
        self.n_assets = len(self.tickers)
        self.risk_free_rate = risk_free_rate

        self.mean_daily_returns = returns_df.mean().values
        self.annualized_returns = self.mean_daily_returns * self.TRADING_DAYS_PER_YEAR
        self.daily_cov_matrix = returns_df.cov().values
        self.annualized_cov_matrix = self.daily_cov_matrix * self.TRADING_DAYS_PER_YEAR

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculates expected return, volatility, and Sharpe ratio for given weights."""
        p_ret = float(np.sum(self.annualized_returns * weights))
        p_vol = float(np.sqrt(weights.T @ self.annualized_cov_matrix @ weights))
        sharpe = (p_ret - self.risk_free_rate) / p_vol if p_vol > 0 else 0.0
        return p_ret, p_vol, sharpe

    def max_sharpe_portfolio(self, short_selling: bool = False) -> Dict[str, object]:
        """Find the portfolio allocation that maximizes the Sharpe Ratio."""
        def neg_sharpe(w):
            _, _, s = self.portfolio_performance(w)
            return -s

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((-1.0, 1.0) if short_selling else (0.0, 1.0) for _ in range(self.n_assets))
        init_guess = np.ones(self.n_assets) / self.n_assets

        res = minimize(
            neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        opt_weights = np.clip(res.x, 0.0, 1.0) if not short_selling else res.x
        opt_weights /= np.sum(opt_weights)
        p_ret, p_vol, sharpe = self.portfolio_performance(opt_weights)

        return {
            "weights": dict(zip(self.tickers, opt_weights.round(4))),
            "weights_array": opt_weights,
            "expected_return": p_ret,
            "volatility": p_vol,
            "sharpe_ratio": sharpe,
            "success": res.success,
        }

    def min_variance_portfolio(self, short_selling: bool = False) -> Dict[str, object]:
        """Find the Global Minimum Variance (GMV) portfolio."""
        constraints = [{
            "type": "eq",
            "fun": lambda w: w.sum() - 1.0,
            "jac": lambda w: np.ones_like(w),
        }]
        bounds = tuple((-1.0, 1.0) if short_selling else (0.0, 1.0) for _ in range(self.n_assets))
        init_guess = np.ones(self.n_assets) / self.n_assets

        res = minimize(
            self._variance,
            init_guess,
            jac=self._variance_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        opt_weights = np.clip(res.x, 0.0, 1.0) if not short_selling else res.x
        opt_weights /= np.sum(opt_weights)
        p_ret, p_vol, sharpe = self.portfolio_performance(opt_weights)

        return {
            "weights": dict(zip(self.tickers, opt_weights.round(4))),
            "weights_array": opt_weights,
            "expected_return": p_ret,
            "volatility": p_vol,
            "sharpe_ratio": sharpe,
            "success": res.success,
        }

    def _variance(self, weights):
        return float(weights @ self.annualized_cov_matrix @ weights)

    def _variance_gradient(self, weights):
        return 2.0 * (self.annualized_cov_matrix @ weights)

    def calculate_efficient_frontier(
        self,
        n_points: int = 50,
        short_selling: bool = False,
    ) -> List[Dict[str, float]]:
        """Generates the Markowitz Efficient Frontier curve points."""
        gmv = self.min_variance_portfolio(short_selling=short_selling)
        min_ret = gmv["expected_return"]
        sorted_returns = np.sort(self.annualized_returns)
        if short_selling:
            short_count = (self.n_assets - 1) // 2
            max_ret = sorted_returns[-(short_count + 1):].sum() - sorted_returns[:short_count].sum()
        else:
            max_ret = sorted_returns[-1]

        if np.isclose(min_ret, max_ret, rtol=0, atol=1e-12):
            return [{
                "return": min_ret,
                "volatility": gmv["volatility"],
                "sharpe_ratio": gmv["sharpe_ratio"],
            }]

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_points = []
        bounds = [(-1.0, 1.0) if short_selling else (0.0, 1.0)] * self.n_assets
        init_guess = gmv["weights_array"]
        constraint_jacobian = np.vstack([np.ones(self.n_assets), self.annualized_returns])

        for target in target_returns:
            constraints = [{
                "type": "eq",
                "fun": lambda w: constraint_jacobian @ w - [1.0, target],
                "jac": lambda w: constraint_jacobian,
            }]
            res = minimize(
                self._variance,
                init_guess,
                jac=self._variance_gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-8, "maxiter": 500},
            )
            if res.success:
                init_guess = res.x
                vol = np.sqrt(res.x.T @ self.annualized_cov_matrix @ res.x)
                s = (target - self.risk_free_rate) / vol if vol > 0 else 0.0
                frontier_points.append({
                    "return": float(target),
                    "volatility": float(vol),
                    "sharpe_ratio": float(s),
                })

        return frontier_points
