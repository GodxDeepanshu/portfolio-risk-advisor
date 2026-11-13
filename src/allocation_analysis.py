"""Validated long-only portfolio inputs and allocation comparisons."""
import numpy as np
import pandas as pd

from src.data_loader import MarketDataLoader
from src.risk_metrics import RiskMetricsCalculator


def parse_holdings(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["symbol", "weight", "comparison_weight", "shock_pct"]
    if not set(required).issubset(frame.columns) or not 1 <= len(frame) <= 30:
        raise ValueError("Provide 1–30 rows with columns: " + ", ".join(required))
    holdings = frame[required].copy()
    symbols = holdings["symbol"].astype("string").str.strip().str.upper()
    if not symbols.str.fullmatch(r"[A-Z0-9][A-Z0-9.^=-]{0,19}", na=False).all() or symbols.duplicated().any():
        raise ValueError("Use valid, unique ticker symbols.")
    holdings["symbol"] = symbols
    for column in required[1:]:
        holdings[column] = pd.to_numeric(holdings[column], errors="coerce")
    if not np.isfinite(holdings[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError("Weights and shocks must be finite numbers.")
    for column in ("weight", "comparison_weight"):
        if (holdings[column] < 0).any() or not np.isclose(holdings[column].sum(), 1, atol=1e-8, rtol=0):
            raise ValueError(f"{column} must contain nonnegative fractions summing to 1.")
    if (holdings["shock_pct"] < -100).any():
        raise ValueError("An asset shock cannot be below −100%.")
    return holdings.set_index("symbol")


def read_prices(frame: pd.DataFrame, symbols: pd.Index) -> pd.DataFrame:
    if "date" not in frame or set(frame.columns) != {"date", *symbols}:
        raise ValueError("Price CSV must contain a date column and exactly the holdings symbols.")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    prices = frame.loc[:, symbols].apply(pd.to_numeric, errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or len(dates) < 3:
        raise ValueError("Provide at least three rows with unique YYYY-MM-DD dates.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Every date must have a finite, positive adjusted close for every symbol.")
    return prices.set_index(dates).sort_index()


def compare_allocations(holdings, prices, capital, risk_free_rate, source):
    returns = MarketDataLoader.compute_daily_returns(prices)
    covariance = returns.cov().to_numpy()
    details = holdings.copy()
    reports = {}
    paths = {}
    for label, column in [("Current", "weight"), ("Comparison", "comparison_weight")]:
        weights = holdings[column].to_numpy()
        report = RiskMetricsCalculator.compute_comprehensive_risk_report(
            returns, weights, capital, risk_free_rate
        )
        weighted_covariance = weights * (covariance @ weights)
        variance = weighted_covariance.sum()
        details[f"{label} share of variance"] = weighted_covariance / variance if variance > 0 else np.nan
        details[f"{label} shock impact"] = weights * holdings["shock_pct"] / 100
        for confidence, minimum in [(95, 100), (99, 500)]:
            if len(returns) < minimum:
                report[f"historical_var_{confidence}"] = np.nan
                report[f"historical_cvar_{confidence}"] = np.nan
        report["scenario_return"] = float(details[f"{label} shock impact"].sum())
        report["concentration_hhi"] = float(weights @ weights)
        reports[label] = report
        paths[label] = (1 + returns @ weights).cumprod()
    payload = {
        "source": source,
        "first_price": prices.index[0].date().isoformat(),
        "last_price": prices.index[-1].date().isoformat(),
        "return_observations": len(returns),
        "currency": "USD",
        "capital": capital,
        "annual_risk_free_rate": risk_free_rate,
        "annualization_days": 252,
        "tail_risk_horizon_days": 1,
        "holdings": holdings.reset_index().to_dict("records"),
        "reports": pd.DataFrame(reports).to_dict(),
    }
    # Convert missing risk estimates to JSON null.
    return pd.DataFrame(reports), details, pd.DataFrame(paths), returns.corr(), pd.Series(payload).to_json(indent=2)
