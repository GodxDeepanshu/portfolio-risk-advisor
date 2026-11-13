"""Streamlit controls and results for the holdings workflow."""
from io import StringIO

import pandas as pd
import streamlit as st

from src.allocation_analysis import compare_allocations, parse_holdings, read_prices
from src.data_loader import MarketDataLoader


EXAMPLE = """symbol,weight,comparison_weight,shock_pct
AAPL,0.50,0.35,-20
GLD,0.30,0.35,5
BND,0.20,0.30,-3
"""


def show_holdings_analysis(capital, risk_free_rate):
    st.header("Holdings and stress tests")
    st.caption("Compare two long-only USD allocations. Enter weights as fractions and shocks as percentages.")
    st.download_button("Download sample holdings", EXAMPLE, "holdings.csv", "text/csv")
    upload = st.file_uploader("Holdings CSV", type="csv")
    price_upload = st.file_uploader(
        "Adjusted price history CSV (optional)", type="csv",
        help="Use a date column and one adjusted close column per ticker, in USD and aligned to consecutive trading sessions.",
    )
    try:
        frame = pd.read_csv(upload if upload is not None else StringIO(EXAMPLE))
        edited = st.data_editor(frame, num_rows="dynamic", key=f"holdings-{upload.file_id if upload else 'sample'}")
        holdings = parse_holdings(edited)
        if price_upload is None:
            prices = MarketDataLoader(holdings.index.tolist()).generate_synthetic_data()
            source = "Synthetic demo prices (seed 42)"
        else:
            prices = read_prices(pd.read_csv(price_upload), holdings.index)
            source = "Uploaded adjusted prices"
        reports, details, wealth, correlation, export = compare_allocations(
            holdings, prices, capital, risk_free_rate, source
        )
    except (ValueError, pd.errors.ParserError) as error:
        st.error(str(error))
        return

    st.caption(f"{source} · {prices.index[0].date()} to {prices.index[-1].date()} · {len(prices) - 1} daily returns")
    st.caption("Historical returns assume daily rebalancing. Stress shocks are hypothetical simultaneous moves.")
    st.subheader("Allocation comparison")
    st.dataframe(reports.style.format("{:,.4g}", na_rep="Unavailable"))
    st.caption("Return and volatility are fractions. One-day VaR and expected shortfall are in USD. Historical estimates require 100 observations at 95% and 500 at 99%.")
    st.subheader("Risk and stress by holding")
    st.dataframe(details.style.format(precision=4, na_rep="Unavailable"))
    st.subheader("Growth of $1")
    st.line_chart(wealth)
    st.subheader("Daily return correlations")
    st.dataframe(correlation.style.format("{:+.2f}", na_rep="Unavailable"))
    st.download_button("Export analysis and assumptions", export, "portfolio-analysis.json", "application/json")
