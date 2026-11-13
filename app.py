"""
Interactive Web Application: AI-Driven Quantitative Portfolio Risk Advisor.
Provides visual portfolio optimization, Monte Carlo simulation paths,
interactive VaR/CVaR risk meters, and NLP sentiment alpha dynamically.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.portfolio_advisor import PortfolioRiskAdvisor
from holdings_view import show_holdings_analysis

st.set_page_config(
    page_title="Quantitative Portfolio Risk Advisor",
    page_icon="📈",
    layout="wide",
)

st.title("Portfolio Risk Advisor")
st.caption("Educational portfolio analysis with historical estimates and hypothetical scenarios.")
view = st.sidebar.radio("Analysis", ["Holdings and stress tests", "Optimization and simulation"])

# Sidebar Controls
st.sidebar.header("⚙️ Simulation & Portfolio Parameters")

portfolio_capital = st.sidebar.number_input(
    "Total Portfolio Capital ($)",
    min_value=10_000.0,
    max_value=100_000_000.0,
    value=1_000_000.0,
    step=50_000.0,
    format="%.2f",
)

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.25,
) / 100.0

if view == "Holdings and stress tests":
    show_holdings_analysis(portfolio_capital, risk_free_rate)
    st.stop()

st.info("This model uses synthetic prices and illustrative sentiment scores.")

mc_iterations = st.sidebar.select_slider(
    "Monte Carlo Iterations",
    options=[1_000, 5_000, 10_000, 20_000],
    value=1_000,
)

time_horizon = st.sidebar.slider(
    "Projection Time Horizon (Trading Days)",
    min_value=30,
    max_value=504,
    value=252,
    step=21,
)

tilt_intensity = st.sidebar.slider(
    "NLP Sentiment Tilt Intensity (%)",
    min_value=0,
    max_value=50,
    value=25,
    step=5,
) / 100.0

# Initialize advisor
advisor = PortfolioRiskAdvisor(
    portfolio_value=portfolio_capital,
    risk_free_rate=risk_free_rate,
    tilt_intensity=tilt_intensity,
)

with st.spinner("Executing quantitative optimization and Monte Carlo simulation..."):
    results = advisor.run_advisory_pipeline(
        mc_iterations=mc_iterations,
        mc_horizon_days=time_horizon,
    )

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
t_rep = results["tilted_risk_report"]
col1.metric("Expected Annual Return", f"{t_rep['annualized_return']*100:.2f}%")
col2.metric("Annualized Volatility", f"{t_rep['annualized_volatility']*100:.2f}%")
col3.metric("Sharpe Ratio", f"{t_rep['sharpe_ratio']:.2f}")
col4.metric("Sortino Ratio", f"{t_rep['sortino_ratio']:.2f}")

st.divider()

# Tab Layout
tab1, tab2, tab3 = st.tabs([
    "📊 Markowitz Efficient Frontier & Allocation",
    "🎲 Monte Carlo Simulation",
    "🛡️ Risk and VaR",
])

with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Efficient Frontier (Risk vs Return)")
        frontier = results["efficient_frontier"]
        f_vols = [p["volatility"] * 100 for p in frontier]
        f_rets = [p["return"] * 100 for p in frontier]

        max_s = results["max_sharpe_portfolio"]
        min_v = results["min_variance_portfolio"]

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)
        ax.plot(f_vols, f_rets, "b-", linewidth=2.5, label="Efficient Frontier")
        ax.scatter(
            [max_s["volatility"] * 100],
            [max_s["expected_return"] * 100],
            color="gold",
            s=180,
            edgecolors="black",
            zorder=5,
            label=f"Max Sharpe ({max_s['sharpe_ratio']:.2f})",
        )
        ax.scatter(
            [min_v["volatility"] * 100],
            [min_v["expected_return"] * 100],
            color="crimson",
            s=140,
            edgecolors="black",
            zorder=5,
            label=f"Global Min Vol ({min_v['volatility']*100:.1f}%)",
        )
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Annualized Expected Return (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Asset Allocations (Base vs Sentiment Tilted)")
        base_w = results["max_sharpe_portfolio"]["weights"]
        tilted_w = results["tilted_weights"]

        alloc_df = pd.DataFrame({
            "Base Weight": base_w,
            "NLP Tilted Weight": tilted_w,
            "Sentiment Signal": results["sentiment_scores"],
        })
        st.dataframe(alloc_df.style.format({
            "Base Weight": "{:.1%}",
            "NLP Tilted Weight": "{:.1%}",
            "Sentiment Signal": "{:+.2f}",
        }), use_container_width=True)

with tab2:
    st.subheader(f"Vectorized Monte Carlo Paths ({mc_iterations:,} Iterations)")
    mc = results["monte_carlo_results"]
    trajectories = mc["trajectories"]
    percentiles = mc["percentiles"]

    fig2, ax2 = plt.subplots(figsize=(10, 4.8), dpi=100)
    days = np.arange(time_horizon + 1)

    # Plot sample trajectories
    sample_indices = np.random.default_rng(42).choice(mc_iterations, size=min(100, mc_iterations), replace=False)
    for idx in sample_indices:
        ax2.plot(days, trajectories[idx], color="gray", alpha=0.08, linewidth=1)

    p50_path = np.percentile(trajectories, 50, axis=0)
    p95_path = np.percentile(trajectories, 95, axis=0)
    p5_path = np.percentile(trajectories, 5, axis=0)

    ax2.plot(days, p50_path, color="blue", linewidth=2.5, label="Median Path (P50)")
    ax2.plot(days, p95_path, color="green", linestyle="--", linewidth=2, label="95th Percentile")
    ax2.plot(days, p5_path, color="red", linestyle="--", linewidth=2, label="5th Percentile (Tail Risk)")

    ax2.axhline(portfolio_capital, color="black", linestyle=":", label="Initial Capital")
    ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Probability of Capital Loss", f"{mc['prob_of_loss']*100:.1f}%")
    col_m2.metric("Median Projected Value", f"${percentiles['median_p50']:,.0f}")
    col_m3.metric("95th Percentile Value", f"${percentiles['p95']:,.0f}")
    col_m4.metric("5th Percentile Tail", f"${percentiles['p5']:,.0f}")

with tab3:
    st.subheader("Parametric & Historical Tail Risk Metrics")
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown("#### Value at Risk (VaR) & Conditional VaR (1-Day)")
        st.write(f"- **Parametric VaR (95%)**: `${t_rep['parametric_var_95']:,.2f}`")
        st.write(f"- **Parametric VaR (99%)**: `${t_rep['parametric_var_99']:,.2f}`")
        st.write(f"- **Historical VaR (95%)**: `${t_rep['historical_var_95']:,.2f}`")
        st.write(f"- **Historical CVaR / Expected Shortfall (95%)**: `${t_rep['historical_cvar_95']:,.2f}`")
        st.write(f"- **Historical CVaR (99%)**: `${t_rep['historical_cvar_99']:,.2f}`")

    with r_col2:
        st.markdown("#### Drawdown & Stability Ratios")
        st.write(f"- **Historical Maximum Drawdown (MDD)**: `{t_rep['maximum_drawdown']*100:.2f}%`")
        st.write(f"- **Calmar Ratio**: `{t_rep['calmar_ratio']:.2f}`")
        st.write(f"- **Annualized Volatility**: `{t_rep['annualized_volatility']*100:.2f}%`")
        st.write(f"- **{time_horizon}-Day Monte Carlo VaR (95%)**: `${mc['mc_var_95']:,.2f}`")

st.caption("Deepanshu Kushwaha | Quantitative Portfolio Risk Advisor Engine")
