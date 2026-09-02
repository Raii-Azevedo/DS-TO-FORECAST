"""Camada de modelagem: Prophet com degradação controlada.

Estratégia em cascata — se um nível falhar, o próximo assume, e a aplicação
nunca fica sem resposta:

1. Prophet com a configuração escolhida.
2. Prophet simplificado (sem sazonalidade, growth linear, `mcmc_samples=0`).
3. Fallback determinístico (regressão linear via OLS) com intervalo empírico.

O nível 3 também é usado direto quando a série é curta demais para o Stan
convergir, evitando a falha de inicialização do modelo.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SETTINGS
from .preprocessing import CleanSeries

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Saída unificada, independente do motor que a produziu."""

    frame: pd.DataFrame          # ds, y, yhat, yhat_lower, yhat_upper, tipo
    engine: str                  # "Prophet", "Prophet (simplificado)" ou "Tendência linear"
    horizon: int
    freq_label: str
    notes: list[str]
    metrics: dict[str, float]

    @property
    def history(self) -> pd.DataFrame:
        return self.frame[self.frame["tipo"] == "Histórico"]

    @property
    def future(self) -> pd.DataFrame:
        return self.frame[self.frame["tipo"] == "Forecast"]


@contextmanager
def _quiet_stan():
    """Silencia os logs verbosos de cmdstanpy/prophet durante o fit."""
    noisy = [logging.getLogger(name) for name in ("cmdstanpy", "prophet", "stan")]
    previous = [lg.level for lg in noisy]
    for lg in noisy:
        lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for lg, level in zip(noisy, previous):
            lg.setLevel(level)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE ignorando zeros no denominador (que gerariam infinito)."""
    mask = np.isfinite(actual) & np.isfinite(predicted) & (np.abs(actual) > 1e-9)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    a = actual.to_numpy(dtype="float64")
    p = predicted.to_numpy(dtype="float64")
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if a.size == 0:
        return {"mape": float("nan"), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    residual = a - p
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return {
        "mape": _safe_mape(a, p),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Motor 3: fallback determinístico
# --------------------------------------------------------------------------- #
def _future_index(last: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    """Gera exatamente ``horizon`` datas futuras a partir de ``last``.

    `pd.date_range(..., inclusive="right")` devolve `horizon` ou `horizon + 1`
    datas conforme a última data esteja ou não alinhada à frequência — uma série
    mensal terminando em 15/03 produzia um array a mais que o de valores, e o
    DataFrame estourava com "All arrays must be of the same length". Somar o
    offset n vezes é determinístico, independente do alinhamento.
    """
    offset = pd.tseries.frequencies.to_offset(freq)
    dates = pd.DatetimeIndex([last + offset * (step + 1) for step in range(horizon)])

    if len(dates) != horizon:  # defesa: nunca deve acontecer
        raise ValueError(f"Esperava {horizon} datas futuras, gerei {len(dates)}.")
    return dates


def _linear_forecast(series: CleanSeries, horizon: int) -> pd.DataFrame:
    """Tendência linear por mínimos quadrados + banda de erro empírica."""
    history = series.df
    x = np.arange(len(history), dtype="float64")
    y = history["y"].to_numpy(dtype="float64")

    if len(history) >= 2 and float(np.std(x)) > 0:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, float(y.mean())

    future_index = np.arange(len(history), len(history) + horizon, dtype="float64")
    fitted = intercept + slope * x
    predicted = intercept + slope * future_index

    residual_std = float(np.std(y - fitted, ddof=0))
    if residual_std == 0 or not np.isfinite(residual_std):
        residual_std = abs(float(np.mean(y))) * 0.05 or 1.0
    band = 1.28 * residual_std  # ~80% de cobertura

    future_dates = _future_index(history["ds"].max(), series.freq, horizon)

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "ds": history["ds"],
                    "y": y,
                    "yhat": fitted,
                    "yhat_lower": fitted - band,
                    "yhat_upper": fitted + band,
                    "tipo": "Histórico",
                }
            ),
            pd.DataFrame(
                {
                    "ds": future_dates,
                    "y": np.nan,
                    "yhat": predicted,
                    "yhat_lower": predicted - band,
                    "yhat_upper": predicted + band,
                    "tipo": "Forecast",
                }
            ),
        ],
        ignore_index=True,
    )


# --------------------------------------------------------------------------- #
# Motores 1 e 2: Prophet
# --------------------------------------------------------------------------- #
def _run_prophet(
    series: CleanSeries,
    horizon: int,
    *,
    growth: str,
    yearly_seasonality: bool,
    weekly_seasonality: bool,
    changepoint_prior_scale: float,
    simplified: bool,
) -> pd.DataFrame:
    from prophet import Prophet  # import tardio: acelera o boot do Streamlit

    history = series.df[["ds", "y"]].copy()
    history["ds"] = pd.to_datetime(history["ds"])
    history["y"] = history["y"].astype("float64")

    # Última barreira antes do Stan: nada de NaN/inf pode passar daqui.
    history = history[np.isfinite(history["y"].to_numpy())].dropna(subset=["ds", "y"])
    if len(history) < SETTINGS.min_points:
        raise ValueError("Pontos válidos insuficientes para o Prophet.")

    model = Prophet(
        growth="linear" if simplified else growth,
        yearly_seasonality=False if simplified else yearly_seasonality,
        weekly_seasonality=False if simplified else weekly_seasonality,
        daily_seasonality=False,
        interval_width=SETTINGS.interval_width,
        changepoint_prior_scale=0.05 if simplified else changepoint_prior_scale,
        mcmc_samples=0,
        uncertainty_samples=200 if simplified else 1000,
    )

    with _quiet_stan():
        model.fit(history)
        future = model.make_future_dataframe(periods=horizon, freq=series.freq)
        forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    if not np.isfinite(result["yhat"].to_numpy()).all():
        raise ValueError("O Prophet retornou valores não finitos.")

    last = history["ds"].max()
    result = result.merge(history, on="ds", how="left")
    result["tipo"] = np.where(result["ds"] <= last, "Histórico", "Forecast")
    return result[["ds", "y", "yhat", "yhat_lower", "yhat_upper", "tipo"]]


# --------------------------------------------------------------------------- #
# Orquestrador
# --------------------------------------------------------------------------- #
def run_forecast(
    series: CleanSeries,
    horizon: int = SETTINGS.default_horizon,
    *,
    growth: str = "linear",
    yearly_seasonality: bool | None = None,
    weekly_seasonality: bool = False,
    changepoint_prior_scale: float = 0.05,
) -> ForecastResult:
    """Executa o forecast com fallback automático. Não levanta exceção."""
    horizon = int(max(1, min(horizon, SETTINGS.max_horizon)))
    notes: list[str] = list(series.warnings)

    if yearly_seasonality is None:
        yearly_seasonality = (
            len(series.df) >= SETTINGS.min_points_yearly_seasonality
            and series.freq in {"D", "W", "MS"}
        )

    if len(series.df) < SETTINGS.min_points_reliable:
        notes.append(
            "Série curta demais para o Prophet convergir com segurança — "
            "usada projeção por tendência linear."
        )
        frame, engine = _linear_forecast(series, horizon), "Tendência linear"
    else:
        attempts = (
            ("Prophet", False),
            ("Prophet (simplificado)", True),
        )
        frame, engine = None, "Tendência linear"
        for label, simplified in attempts:
            try:
                frame = _run_prophet(
                    series,
                    horizon,
                    growth=growth,
                    yearly_seasonality=bool(yearly_seasonality),
                    weekly_seasonality=weekly_seasonality,
                    changepoint_prior_scale=changepoint_prior_scale,
                    simplified=simplified,
                )
                engine = label
                if simplified:
                    notes.append(
                        "O modelo completo não convergiu; usada configuração "
                        "simplificada (sem sazonalidade)."
                    )
                break
            except Exception as exc:  # noqa: BLE001 - qualquer falha vira fallback
                logger.warning("Falha no motor %s: %s", label, exc)

        if frame is None:
            notes.append(
                "O Prophet não convergiu com estes dados — usada projeção por "
                "tendência linear como alternativa."
            )
            frame = _linear_forecast(series, horizon)

    frame = frame.sort_values("ds").reset_index(drop=True)
    history = frame[frame["tipo"] == "Histórico"]
    metrics = _compute_metrics(history["y"], history["yhat"])

    return ForecastResult(
        frame=frame,
        engine=engine,
        horizon=horizon,
        freq_label=series.freq_label,
        notes=notes,
        metrics=metrics,
    )
