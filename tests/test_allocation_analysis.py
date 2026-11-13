import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.allocation_analysis import compare_allocations, parse_holdings, read_prices
from src.monte_carlo import MonteCarloEngine


def test_allocation_risk_and_export():
    holdings = parse_holdings(pd.DataFrame({
        "symbol": ["A", "B"],
        "weight": [1, 0],
        "comparison_weight": [0.5, 0.5],
        "shock_pct": [-20, 10],
    }))
    prices = read_prices(pd.DataFrame({
        "date": ["2025-01-06", "2025-01-07", "2025-01-08"],
        "A": [100, 80, 100],
        "B": [100, 100, 100],
    }), holdings.index)
    report, details, _, _, export = compare_allocations(holdings, prices, 100, 0, "Test")
    assert report.loc["scenario_return"].tolist() == pytest.approx([-0.20, -0.05])
    assert report.loc["maximum_drawdown", "Current"] == pytest.approx(0.20)
    assert details["Comparison share of variance"].tolist() == pytest.approx([1, 0])
    assert json.loads(export)["reports"]["Current"]["historical_var_99"] is None
    with pytest.raises(ValueError, match="finite, positive"):
        read_prices(prices.reset_index().rename(columns={"index": "date"}).assign(A=0), holdings.index)
    with pytest.raises(ValueError, match="summing to 1"):
        parse_holdings(holdings.reset_index().assign(weight=0))


def test_simulation_supports_singular_covariance():
    model = MonteCarloEngine([0, 0], np.zeros((2, 2)), [0.5, 0.5], 100)
    paths = model.run_simulations(iterations=5, time_horizon_days=2)["trajectories"]
    np.testing.assert_array_equal(paths, np.full((5, 3), 100))


def test_dashboard_modes_render():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 4
    app.sidebar.radio[0].set_value("Optimization and simulation").run()
    assert not app.exception
    assert len(app.metric) == 8
