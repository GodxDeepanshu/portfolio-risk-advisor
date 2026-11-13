"""
Risk Metrics Engine: Institutional Parametric & Historical Risk Calculations.
Includes Value at Risk (VaR 95%/99%), Conditional VaR (CVaR/Expected Shortfall),
Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and Covariance Rebalancing matrices.
"""
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm


class RiskMetricsCalculator:
    """Computes parametric and historical quantitative risk metrics."""

    TRADING_DAYS_PER_YEAR = 252

    @classmethod
    def portfolio_return(cls, weights: np.ndarray, mean_returns: np.ndarray) -> float:
        """Annualized expected portfolio return."""
        return float(np.sum(mean_returns * weights) * cls.TRADING_DAYS_PER_YEAR)

    @classmethod
    def portfolio_volatility(cls, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Annualized portfolio standard deviation (volatility)."""
        variance = weights.T @ cov_matrix @ weights
        return float(np.sqrt(variance * cls.TRADING_DAYS_PER_YEAR))

    @classmethod
    def sharpe_ratio(
        cls,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.04,
    ) -> float:
        """Annualized Sharpe Ratio."""
        p_ret = cls.portfolio_return(weights, mean_returns)
        p_vol = cls.portfolio_volatility(weights, cov_matrix)
        if p_vol == 0:
            return 0.0
        return (p_ret - risk_free_rate) / p_vol

    @classmethod
    def sortino_ratio(
        cls,
        portfolio_returns: pd.Series,
        risk_free_rate: float = 0.04,
        target_return: float = 0.0,
    ) -> float:
        """Annualized Sortino Ratio focusing on downside volatility."""
        mean_annual_return = portfolio_returns.mean() * cls.TRADING_DAYS_PER_YEAR
        shortfalls = np.minimum(portfolio_returns - target_return, 0)
        downside_deviation = np.sqrt((shortfalls**2).mean() * cls.TRADING_DAYS_PER_YEAR)
        if downside_deviation == 0:
            return np.nan
        return float((mean_annual_return - risk_free_rate) / downside_deviation)

    @classmethod
    def parametric_var(
        cls,
        portfolio_value: float,
        daily_mean: float,
        daily_std: float,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Parametric (Variance-Covariance) Value at Risk.
        VaR = Portfolio_Value * (Z_alpha * daily_std * sqrt(horizon) - daily_mean * horizon)
        """
        z_score = norm.ppf(confidence_level)
        var_pct = z_score * daily_std * np.sqrt(horizon_days) - (daily_mean * horizon_days)
        return float(portfolio_value * max(var_pct, 0.0))

    @classmethod
    def historical_var_cvar(
        cls,
        portfolio_returns: pd.Series,
        portfolio_value: float = 1_000_000.0,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Historical Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall).
        """
        alpha_pct = (1.0 - confidence_level) * 100.0
        cutoff_return = np.percentile(portfolio_returns, alpha_pct)
        var_loss = float(-cutoff_return * portfolio_value)

        # CVaR is the expectation of losses exceeding the VaR threshold
        tail_losses = portfolio_returns[portfolio_returns <= cutoff_return]
        if len(tail_losses) > 0:
            cvar_loss = float(-tail_losses.mean() * portfolio_value)
        else:
            cvar_loss = var_loss

        return max(var_loss, 0.0), max(cvar_loss, 0.0)

    @classmethod
    def maximum_drawdown(cls, price_or_cum_series: Union[pd.Series, np.ndarray]) -> Tuple[float, pd.Series]:
        """
        Computes maximum drawdown percentage and historical drawdown trajectory.
        """
        if isinstance(price_or_cum_series, np.ndarray):
            series = pd.Series(price_or_cum_series)
        else:
            series = price_or_cum_series.copy()

        cum_peaks = series.cummax()
        drawdown_series = (series - cum_peaks) / cum_peaks
        max_dd = float(drawdown_series.min())
        return abs(max_dd), drawdown_series

    @classmethod
    def compute_comprehensive_risk_report(
        cls,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float = 1_000_000.0,
        risk_free_rate: float = 0.04,
    ) -> Dict[str, float]:
        """Generates an institutional risk scorecard for a portfolio allocation."""
        daily_portfolio_returns = returns_df.dot(weights)
        daily_mean = daily_portfolio_returns.mean()
        daily_std = daily_portfolio_returns.std()
        cov_matrix = returns_df.cov().values
        mean_returns = returns_df.mean().values

        # VaR metrics
        p_var_95 = cls.parametric_var(portfolio_value, daily_mean, daily_std, confidence_level=0.95)
        p_var_99 = cls.parametric_var(portfolio_value, daily_mean, daily_std, confidence_level=0.99)
        h_var_95, h_cvar_95 = cls.historical_var_cvar(daily_portfolio_returns, portfolio_value, 0.95)
        h_var_99, h_cvar_99 = cls.historical_var_cvar(daily_portfolio_returns, portfolio_value, 0.99)

        # Drawdown
        cum_returns = (1.0 + daily_portfolio_returns).cumprod()
        mdd, _ = cls.maximum_drawdown(np.r_[1.0, cum_returns.to_numpy()])

        # Performance Ratios
        ann_return = cls.portfolio_return(weights, mean_returns)
        ann_vol = cls.portfolio_volatility(weights, cov_matrix)
        sharpe = cls.sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
        sortino = cls.sortino_ratio(daily_portfolio_returns, risk_free_rate)
        calmar = (ann_return - risk_free_rate) / mdd if mdd > 0 else 0.0

        return {
            "portfolio_value": portfolio_value,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "maximum_drawdown": mdd,
            "calmar_ratio": calmar,
            "parametric_var_95": p_var_95,
            "parametric_var_99": p_var_99,
            "historical_var_95": h_var_95,
            "historical_cvar_95": h_cvar_95,
            "historical_var_99": h_var_99,
            "historical_cvar_99": h_cvar_99,
        }
