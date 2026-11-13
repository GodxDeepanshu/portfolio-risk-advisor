# Portfolio Risk Advisor

A Streamlit dashboard for comparing portfolio risk, running stress tests, and
exploring long-only Markowitz allocations and Monte Carlo scenarios.

## Run locally

Use Python 3.10 or later. Create a virtual environment with
`python -m venv .venv` and activate it before installing dependencies.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m pytest -q
```

You can also start the dashboard with `docker compose up --build`.

## Try the dashboard

1. Open **Holdings and stress tests** and inspect the preloaded allocation.
2. Edit the current weights, comparison weights, and hypothetical shocks.
   You can also download the sample holdings and upload an edited CSV.
3. Review each portfolio's risk metrics and the holdings driving the scenario.
   Download the JSON report for the input weights, shocks, settings, date
   coverage, and results. Keep the original prices to reproduce uploaded runs.
4. Open **Optimization and simulation** to explore the efficient frontier and
   simulated terminal wealth with synthetic prices and illustrative sentiment.

The holdings CSV columns are `symbol,weight,comparison_weight,shock_pct`.
Each weight column must sum to one. A weight of 0.25 means 25% and a shock of
-20 means a 20% price decline. Up to 30 unique symbols are supported.

Without uploaded prices the dashboard uses seeded synthetic data. For your own
history, upload a wide CSV with a `date` column in YYYY-MM-DD format and one
column per holdings symbol. Supply complete, positive, USD-denominated adjusted
closing prices for consecutive common trading sessions. Date coverage is shown
in the dashboard. No network data provider or API key is required.

## Methodology

- Historical paths compound daily simple returns at constant portfolio weights,
  implying daily rebalancing with zero costs. Arithmetic mean returns and
  volatility are annualized using 252 sessions.
- Concentration is the sum of squared weights. A holding's share of portfolio
  variance is its weight times its covariance with the portfolio, divided by
  portfolio variance. Negative shares indicate diversification effects.
- Stress results sum weight times price shock. Drawdown compares portfolio
  wealth with its running peak, including initial capital.
- One-day parametric VaR uses a normal-return model. Historical VaR uses the
  lower return percentile and expected shortfall averages returns at or below
  the threshold. Dollar loss estimates are floored at zero. The holdings view
  requires at least 100 observations for 95% historical estimates and 500 for
  99% estimates. Threshold ties can increase the number of tail observations.
- Simulations draw daily fixed-weight portfolio returns from a normal model
  using the estimated weighted mean and covariance. Daily losses are capped at
  100% before compounding. Simulations assume independent days and fixed
  parameters. Sentiment scores in the dashboard are illustrative.
- Uploaded prices are analyzed as supplied. The dashboard checks completeness
  within each row and relies on the uploader to supply consecutive sessions,
  corporate-action adjustments, and consistent currency. Historical estimates
  and hypothetical scenarios are intended for educational analysis.
