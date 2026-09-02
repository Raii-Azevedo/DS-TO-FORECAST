"""Testes do pipeline de saneamento e forecast.

Foco: garantir que nenhum dado sujo chegue ao Prophet como NaN/inf — a causa
do erro `normal_lpdf: Random variable is nan` que derrubava a aplicação.

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    profile_columns,
    suggest_date_column,
    suggest_value_column,
    usable_date_columns,
)
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


# --------------------------------------------------------------------------- #
# Detecção de colunas em base transacional larga
# --------------------------------------------------------------------------- #
def _wide_frame(n: int = 800) -> pd.DataFrame:
    """Imita uma extração de BI: códigos, checkbox, IDs e valores em texto BR."""
    rs = np.random.RandomState(7)
    return pd.DataFrame(
        {
            "[fn] Usuario": ["jose@exemplo.com"] * n,
            "CC Filtro Prevista Entrega": rs.choice([True, False], n),
            "Cd Business Unit": ["BU02"] * n,
            "Cd Centro Distribuicao": [f"BR0140{i % 99:02d}" for i in range(n)],
            "Cd Item Oc Pedido": rs.randint(1000, 400000, n),
            "Dt Prevista Entrega": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rs.randint(0, 400, n), "D"),
            "Data Faturamento": (
                pd.to_datetime("2024-01-01") + pd.to_timedelta(rs.randint(0, 400, n), "D")
            ).strftime("%d/%m/%Y"),
            # formato brasileiro: "R$ 12.345,50"
            "Pedido Faturado": [
                "R$ " + f"{v:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")
                for v in rs.randint(500, 90000, n)
            ],
            "Observacao": [None] * n,
        }
    )


@pytest.mark.parametrize("code", ["BU02", "BR014074", "jose@exemplo.com", "SP-01"])
def test_codes_never_become_numbers(code):
    """Regressão: códigos alfanuméricos viravam número ao remover as letras."""
    assert pd.isna(_parse_number(code))


def test_boolean_column_is_not_a_date():
    """Regressão: uma coluna de checkbox era sugerida como coluna de data."""
    profile = profile_columns(_wide_frame())
    assert profile.loc["CC Filtro Prevista Entrega", "date_ratio"] == 0.0
    assert "CC Filtro Prevista Entrega" not in usable_date_columns(profile)


def test_id_column_is_not_a_date():
    """Regressão: um ID inteiro era lido como epoch em nanossegundos."""
    profile = profile_columns(_wide_frame())
    assert profile.loc["Cd Item Oc Pedido", "date_ratio"] == 0.0


def test_datetime_column_is_not_a_metric():
    """Regressão: uma coluna de data virava métrica via nanossegundos."""
    profile = profile_columns(_wide_frame())
    assert profile.loc["Dt Prevista Entrega", "value_ratio"] == 0.0


def test_suggestion_picks_the_business_columns():
    df = _wide_frame()
    profile = profile_columns(df)
    date_col = suggest_date_column(df, profile)
    value_col = suggest_value_column(df, exclude=date_col, profile=profile)
    assert date_col == "Data Faturamento"
    assert value_col == "Pedido Faturado"


def test_wide_frame_produces_a_forecast():
    df = _wide_frame()
    profile = profile_columns(df)
    date_col = suggest_date_column(df, profile)
    value_col = suggest_value_column(df, exclude=date_col, profile=profile)
    result = run_forecast(prepare_series(df, date_col, value_col, aggregation="sum"), horizon=6)
    assert len(result.future) == 6
    assert np.isfinite(result.future["yhat"].to_numpy()).all()


# --------------------------------------------------------------------------- #
# CSS injetado no Streamlit
# --------------------------------------------------------------------------- #
def test_css_survives_the_markdown_parser():
    """Regressão: o CSS aparecia como texto na tela.

    O Streamlit passa `st.markdown` por um parser CommonMark. Linhas indentadas
    com 4+ espaços viram bloco de código e uma linha em branco encerra o bloco
    HTML — nos dois casos a folha de estilo vaza como texto visível.
    """
    from src.theme import _css

    css = _css()
    assert css.startswith("<style>") and css.rstrip().endswith("</style>")
    for line in css.splitlines():
        assert line.strip(), "linha em branco encerraria o bloco HTML"
        assert not line.startswith("    "), "indentação viraria bloco de código"


def test_metrics_handle_zero_values():
    df = pd.DataFrame(
        {
            "Data": pd.date_range("2023-01-01", periods=15, freq="MS"),
            "Valor": [0, 10, 0, 20, 30, 0, 40, 50, 60, 0, 70, 80, 90, 0, 100],
        }
    )
    result = run_forecast(prepare_series(df, "Data", "Valor"), horizon=3)
    assert np.isfinite(result.metrics["mape"]), "MAPE não pode virar infinito com zeros"
