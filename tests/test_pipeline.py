"""Testes do pipeline de saneamento e forecast.

Foco: garantir que nenhum dado sujo chegue ao Prophet como NaN/inf — a causa
do erro `normal_lpdf: Random variable is nan` que derrubava a aplicação.

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import run_forecast
from src.preprocessing import (
    ValidationError,
    _parse_number,
    infer_frequency,
    prepare_series,
    to_numeric,
)


# --------------------------------------------------------------------------- #
# Conversão numérica
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1234.56, 1234.56),
        ("1234.56", 1234.56),
        ("1.234,56", 1234.56),
        ("1,234.56", 1234.56),
        ("R$ 1.234,56", 1234.56),
        ("(1.234,56)", -1234.56),
        ("12,5", 12.5),
        ("1,234", 1234.0),
        ("45%", 45.0),
        ("", np.nan),
        ("-", np.nan),
        ("N/A", np.nan),
        ("#DIV/0!", np.nan),
        (None, np.nan),
        (float("inf"), np.nan),
    ],
)
def test_parse_number(raw, expected):
    result = _parse_number(raw)
    if pd.isna(expected):
        assert pd.isna(result)
    else:
        assert result == pytest.approx(expected)


def test_to_numeric_never_leaves_non_finite():
    series = pd.Series(["10", "abc", np.inf, -np.inf, None, "R$ 20,50"])
    converted = to_numeric(series)
    finite = converted.dropna()
    assert np.isfinite(finite.to_numpy()).all()
    assert finite.tolist() == [10.0, 20.5]


# --------------------------------------------------------------------------- #
# Saneamento da série
# --------------------------------------------------------------------------- #
def _messy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Data": [
                "01/01/2024", "01/02/2024", "data ruim", "01/03/2024",
                "01/04/2024", "01/04/2024", "01/05/2024", "01/06/2024",
                "01/07/2024", "01/08/2024", None, "01/09/2024",
            ],
            "Valor": [
                "R$ 1.000,00", "1.200,50", "999", "N/A", "700,00",
                "800,00", "1.700,25", "(200,00)", "1.900", "2.000,00",
                "50", 2100.0,
            ],
        }
    )


def test_prepare_series_removes_nan_and_aggregates():
    series = prepare_series(_messy_frame(), "Data", "Valor")
    values = series.df["y"].to_numpy()

    assert np.isfinite(values).all(), "nenhum NaN/inf pode sobreviver à limpeza"
    assert series.df["ds"].is_monotonic_increasing
    assert not series.df["ds"].duplicated().any()
    assert series.dropped_invalid_date >= 1
    assert series.dropped_invalid_value >= 1
    assert series.duplicates_aggregated >= 1
    assert series.freq == "MS"


def test_prepare_series_rejects_all_nan_column():
    df = pd.DataFrame({"Data": ["01/01/2024", "01/02/2024"], "Valor": [None, "xyz"]})
    with pytest.raises(ValidationError, match="Nenhuma linha válida"):
        prepare_series(df, "Data", "Valor")


def test_prepare_series_rejects_same_column():
    df = pd.DataFrame({"Data": ["01/01/2024", "01/02/2024"], "Valor": [1, 2]})
    with pytest.raises(ValidationError):
        prepare_series(df, "Data", "Data")


def test_prepare_series_rejects_missing_column():
    df = pd.DataFrame({"Data": ["01/01/2024"], "Valor": [1]})
    with pytest.raises(ValidationError, match="não existe"):
        prepare_series(df, "Data", "Inexistente")


def test_infer_frequency_daily_and_monthly():
    assert infer_frequency(pd.Series(pd.date_range("2024-01-01", periods=10, freq="D")))[0] == "D"
    assert infer_frequency(pd.Series(pd.date_range("2024-01-01", periods=10, freq="MS")))[0] == "MS"


# --------------------------------------------------------------------------- #
# Forecast: nunca deve levantar exceção
# --------------------------------------------------------------------------- #
def test_forecast_on_messy_data():
    result = run_forecast(prepare_series(_messy_frame(), "Data", "Valor"), horizon=6)
    assert len(result.future) == 6
    assert np.isfinite(result.future["yhat"].to_numpy()).all()
    assert (result.future["yhat_lower"] <= result.future["yhat_upper"]).all()


def test_forecast_on_short_series_uses_fallback():
    df = pd.DataFrame(
        {"Data": ["01/01/2024", "01/02/2024", "01/03/2024"], "Valor": [100, 110, 120]}
    )
    result = run_forecast(prepare_series(df, "Data", "Valor"), horizon=3)
    assert result.engine == "Tendência linear"
    assert np.isfinite(result.future["yhat"].to_numpy()).all()


def test_forecast_on_constant_series():
    df = pd.DataFrame(
        {
            "Data": pd.date_range("2024-01-01", periods=18, freq="MS"),
            "Valor": [500.0] * 18,
        }
    )
    result = run_forecast(prepare_series(df, "Data", "Valor"), horizon=4)
    assert np.isfinite(result.future["yhat"].to_numpy()).all()


def test_forecast_horizon_is_capped():
    df = pd.DataFrame(
        {
            "Data": pd.date_range("2022-01-01", periods=36, freq="MS"),
            "Valor": np.linspace(100, 400, 36),
        }
    )
    result = run_forecast(prepare_series(df, "Data", "Valor"), horizon=999)
    assert result.horizon <= 36


def test_metrics_handle_zero_values():
    df = pd.DataFrame(
        {
            "Data": pd.date_range("2023-01-01", periods=15, freq="MS"),
            "Valor": [0, 10, 0, 20, 30, 0, 40, 50, 60, 0, 70, 80, 90, 0, 100],
        }
    )
    result = run_forecast(prepare_series(df, "Data", "Valor"), horizon=3)
    assert np.isfinite(result.metrics["mape"]), "MAPE não pode virar infinito com zeros"
