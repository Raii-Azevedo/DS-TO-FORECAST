"""Gráficos Plotly com a paleta Artefact."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .config import BRAND
from .forecasting import ForecastResult


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(fig: go.Figure, title: str, y_title: str = "Valor") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=BRAND.navy, family=BRAND.font)),
        font=dict(family=BRAND.font, color=BRAND.ink, size=12),
        plot_bgcolor=BRAND.surface,
        paper_bgcolor=BRAND.surface,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=54, b=10),
        height=440,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(title="", showgrid=False, linecolor=BRAND.line, ticks="outside"),
        yaxis=dict(
            title=y_title, gridcolor=BRAND.line, zerolinecolor=BRAND.line,
            tickformat=",.0f",
        ),
    )
    return fig


def forecast_chart(result: ForecastResult, y_title: str = "Valor") -> go.Figure:
    """Histórico, projeção e faixa de confiança em um único gráfico."""
    history, future = result.history, result.future

    # Conecta as duas linhas no ponto de corte, evitando o "buraco" visual.
    bridge = history.tail(1)
    joined = pd.concat([bridge, future], ignore_index=True) if not history.empty else future

    fig = go.Figure()

    if not joined.empty:
        fig.add_trace(go.Scatter(
            x=joined["ds"], y=joined["yhat_upper"], mode="lines",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=joined["ds"], y=joined["yhat_lower"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor=_hex_to_rgba(BRAND.pink, 0.13),
            name="Intervalo de confiança", hoverinfo="skip",
        ))

    if not history.empty:
        fig.add_trace(go.Scatter(
            x=history["ds"], y=history["y"], mode="lines",
            name="Histórico",
            line=dict(color=BRAND.navy, width=2.5),
            hovertemplate="%{x|%b/%Y}<br>%{y:,.2f}<extra>Histórico</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=history["ds"], y=history["yhat"], mode="lines",
            name="Ajuste do modelo",
            line=dict(color=BRAND.violet, width=1.4, dash="dot"),
            hovertemplate="%{x|%b/%Y}<br>%{y:,.2f}<extra>Ajuste</extra>",
        ))

    if not joined.empty:
        fig.add_trace(go.Scatter(
            x=joined["ds"], y=joined["yhat"], mode="lines+markers",
            name="Projeção",
            line=dict(color=BRAND.pink, width=2.5),
            marker=dict(size=6, color=BRAND.pink),
            hovertemplate="%{x|%b/%Y}<br>%{y:,.2f}<extra>Projeção</extra>",
        ))

    if not history.empty and not future.empty:
        fig.add_vline(
            x=history["ds"].max(), line_width=1,
            line_dash="dash", line_color=BRAND.muted,
        )

    return _base_layout(
        fig,
        f"Projeção de {result.horizon} período(s) · frequência {result.freq_label.lower()}",
        y_title,
    )


def residuals_chart(result: ForecastResult) -> go.Figure:
    """Erro do modelo (real − ajustado) no período histórico."""
    history = result.history.copy()
    history["residuo"] = history["y"] - history["yhat"]
    colors = [BRAND.cyan if v >= 0 else BRAND.amber for v in history["residuo"]]

    fig = go.Figure(
        go.Bar(
            x=history["ds"], y=history["residuo"],
            marker_color=colors, name="Resíduo",
            hovertemplate="%{x|%b/%Y}<br>%{y:,.2f}<extra>Resíduo</extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=BRAND.muted)
    fig = _base_layout(fig, "Resíduos do ajuste histórico", "Real − Previsto")
    fig.update_layout(height=280, showlegend=False)
    return fig
