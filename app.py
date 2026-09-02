"""P&L Forecast — aplicação Streamlit de previsão de séries financeiras.

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.charts import forecast_chart, residuals_chart
from src.config import APP, FREQUENCIES, SETTINGS
from src.data_loader import (
    USABLE_THRESHOLD,
    DataLoadError,
    excel_sheet_names,
    load_dataframe,
    profile_columns,
    suggest_date_column,
    suggest_value_column,
    usable_date_columns,
    usable_value_columns,
)
from src.forecasting import ForecastResult, run_forecast
from src.preprocessing import CleanSeries, ValidationError, prepare_series
from src.theme import hero, inject_css, kpi, section

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title=f"{APP.title} · {APP.owner}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Camada com cache — evita reprocessar a cada interação do widget
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load(raw: bytes, filename: str, sheet: str | int) -> pd.DataFrame:
    return load_dataframe(raw, filename, sheet)


@st.cache_data(show_spinner="Analisando as colunas...")
def _profile(df: pd.DataFrame) -> pd.DataFrame:
    return profile_columns(df)


@st.cache_data(show_spinner=False)
def _prepare(
    df: pd.DataFrame, date_col: str, value_col: str, agg: str, freq: str | None
) -> CleanSeries:
    return prepare_series(df, date_col, value_col, aggregation=agg, freq_override=freq)


@st.cache_data(show_spinner="Treinando o modelo...")
def _forecast(
    series: CleanSeries,
    horizon: int,
    growth: str,
    yearly: bool | None,
    weekly: bool,
    changepoint: float,
) -> ForecastResult:
    return run_forecast(
        series,
        horizon,
        growth=growth,
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        changepoint_prior_scale=changepoint,
    )


def _fmt(value: float, decimals: int = 2) -> str:
    """Formata número no padrão brasileiro (1.234,56)."""
    if value is None or not pd.notna(value):
        return "—"
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


# --------------------------------------------------------------------------- #
# Aplicação
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()
    hero()

    with st.sidebar:
        st.markdown("## Base de dados")
        uploaded = st.file_uploader(
            "Arraste um arquivo",
            type=[e.lstrip(".") for e in APP.supported_extensions],
            help="CSV, XLSX ou XLS com uma coluna de data e uma de valores.",
        )

    if uploaded is None:
        _welcome()
        return

    raw = uploaded.getvalue()

    # ---- Aba do Excel ---------------------------------------------------- #
    sheet: str | int = 0
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        sheets = excel_sheet_names(raw)
        if len(sheets) > 1:
            sheet = st.sidebar.selectbox("Aba da planilha", sheets)

    # ---- Leitura --------------------------------------------------------- #
    try:
        data = _load(raw, uploaded.name, sheet)
    except DataLoadError as exc:
        st.error(f"**Não foi possível ler o arquivo.** {exc}")
        return

    # ---- Mapeamento de colunas ------------------------------------------- #
    profile = _profile(data)
    date_options = usable_date_columns(profile)
    value_options = usable_value_columns(profile)

    if not date_options or not value_options:
        _render_mapping_error(data, profile, date_options, value_options)
        return

    def _label(column: str, kind: str) -> str:
        ratio = profile.loc[column, f"{kind}_ratio"]
        return f"{column} · {ratio:.0%}"

    with st.sidebar:
        st.markdown("## Mapeamento")
        st.caption(
            f"{len(date_options)} coluna(s) servem como data e "
            f"{len(value_options)} como valor. O percentual indica quantas "
            "linhas convertem corretamente."
        )
        show_all = st.checkbox(
            "Mostrar todas as colunas", value=False,
            help="Por padrão só aparecem as colunas que realmente convertem.",
        )

        date_list = list(data.columns) if show_all else date_options
        date_col = st.selectbox(
            "Coluna de data", date_list,
            index=date_list.index(suggest_date_column(data, profile))
            if suggest_date_column(data, profile) in date_list else 0,
            format_func=lambda c: _label(c, "date"),
        )

        remaining = usable_value_columns(profile, exclude=date_col)
        value_list = (
            [c for c in data.columns if c != date_col] if show_all else remaining
        )
        if not value_list:
            st.error("Nenhuma coluna de valor sobrou. Escolha outra coluna de data.")
            return
        suggested = suggest_value_column(data, exclude=date_col, profile=profile)
        value_col = st.selectbox(
            "Coluna de valor", value_list,
            index=value_list.index(suggested) if suggested in value_list else 0,
            format_func=lambda c: _label(c, "value"),
        )
        aggregation = st.selectbox(
            "Datas repetidas", ["sum", "mean", "median", "last", "max", "min"],
            format_func=lambda v: {
                "sum": "Somar", "mean": "Média", "median": "Mediana",
                "last": "Último valor", "max": "Máximo", "min": "Mínimo",
            }[v],
            help="Como consolidar múltiplas linhas com a mesma data.",
        )

    # ---- Saneamento ------------------------------------------------------ #
    try:
        series = _prepare(data, date_col, value_col, aggregation, None)
    except ValidationError as exc:
        st.error(f"**Dados inválidos para forecast.** {exc}")
        _render_column_hints(profile, date_options, value_options)
        with st.expander("Ver amostra do arquivo"):
            st.dataframe(data.head(20), use_container_width=True)
        return

    # ---- Parâmetros do modelo -------------------------------------------- #
    with st.sidebar:
        st.markdown("## Modelo")
        st.caption(f"Frequência detectada: **{series.freq_label}**")
        freq_label = st.selectbox(
            "Frequência", list(FREQUENCIES),
            index=list(FREQUENCIES.values()).index(series.freq)
            if series.freq in FREQUENCIES.values() else 2,
        )
        horizon = st.slider(
            "Períodos a prever", 1, SETTINGS.max_horizon, SETTINGS.default_horizon
        )
        with st.expander("Opções avançadas"):
            growth = st.selectbox(
                "Tendência", ["linear", "flat"],
                format_func=lambda v: {"linear": "Linear", "flat": "Plana"}[v],
            )
            seasonality = st.selectbox(
                "Sazonalidade anual", ["Automática", "Ativada", "Desativada"]
            )
            changepoint = st.slider(
                "Flexibilidade da tendência", 0.01, 0.50, 0.05, 0.01,
                help="Valores maiores permitem que a tendência mude mais rápido.",
            )

    if FREQUENCIES[freq_label] != series.freq:
        try:
            series = _prepare(
                data, date_col, value_col, aggregation, FREQUENCIES[freq_label]
            )
        except ValidationError as exc:
            st.error(str(exc))
            return

    yearly = {"Automática": None, "Ativada": True, "Desativada": False}[seasonality]
    result = _forecast(series, horizon, growth, yearly, False, changepoint)

    # ---- Saída ----------------------------------------------------------- #
    _render_quality(series, result)
    _render_kpis(series, result)

    section("Projeção", f"Motor: {result.engine} · intervalo de {int(SETTINGS.interval_width * 100)}%")
    st.plotly_chart(forecast_chart(result, value_col), use_container_width=True)

    _render_tables(result, series, value_col)


def _welcome() -> None:
    """Tela inicial quando ainda não há arquivo."""
    section("Como usar", "Três passos até a projeção")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(
            """
            1. **Carregue a base** na barra lateral (CSV, XLSX ou XLS).
            2. **Confirme as colunas** de data e de valor — elas são detectadas
               automaticamente, mas você pode trocá-las.
            3. **Escolha o horizonte** e leia a projeção, o intervalo de
               confiança e as métricas de qualidade do ajuste.

            A base é limpa antes da modelagem: datas inválidas e valores não
            numéricos são descartados, datas repetidas são consolidadas e
            formatos brasileiros (`R$ 1.234,56`, negativos entre parênteses)
            são convertidos automaticamente.
            """
        )
    with right:
        st.markdown("**Formato esperado**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Data": ["01/01/2025", "01/02/2025", "01/03/2025"],
                    "Valor": [120000.00, 138500.50, 141200.75],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_quality(series: CleanSeries, result: ForecastResult) -> None:
    """Avisos de limpeza e degradação do modelo."""
    if not result.notes:
        return
    section("Qualidade dos dados", f"{series.rows_out} de {series.rows_in} linhas utilizadas")
    for note in result.notes:
        st.warning(note, icon="⚠️")


def _render_kpis(series: CleanSeries, result: ForecastResult) -> None:
    section("Resumo", "Indicadores do ajuste e da projeção")
    future = result.future
    mape = result.metrics["mape"]

    accuracy = "—" if pd.isna(mape) else f"{_fmt(100 - mape, 1)}%"
    quality = (
        "—" if pd.isna(mape)
        else "Excelente" if mape < 10
        else "Boa" if mape < 20
        else "Razoável" if mape < 35
        else "Baixa — revise os dados"
    )

    columns = st.columns(4)
    with columns[0]:
        kpi(
            "Histórico",
            f"{series.rows_out}",
            f"{series.start:%b/%Y} → {series.end:%b/%Y}",
        )
    with columns[1]:
        kpi(
            "Acurácia do ajuste",
            accuracy,
            f"MAPE {_fmt(mape, 1)}% · {quality}",
        )
    with columns[2]:
        total = float(future["yhat"].sum()) if not future.empty else float("nan")
        kpi("Total projetado", _fmt(total), f"{result.horizon} período(s) à frente")
    with columns[3]:
        if future.empty or series.df.empty:
            kpi("Variação esperada", "—", "")
        else:
            last = float(series.df["y"].iloc[-1])
            nxt = float(future["yhat"].iloc[-1])
            delta = (nxt / last - 1) * 100 if abs(last) > 1e-9 else float("nan")
            kpi(
                "Variação esperada",
                "—" if pd.isna(delta) else f"{'+' if delta >= 0 else ''}{_fmt(delta, 1)}%",
                "último real → último previsto",
            )


def _render_tables(result: ForecastResult, series: CleanSeries, value_col: str) -> None:
    section("Detalhamento", "Valores período a período")
    tab_forecast, tab_full, tab_diag = st.tabs(
        ["Projeção", "Série completa", "Diagnóstico"]
    )

    column_config = {
        "Período": st.column_config.DatetimeColumn(format="DD/MM/YYYY"),
        "Previsto": st.column_config.NumberColumn(format="%.2f"),
        "Mínimo": st.column_config.NumberColumn(format="%.2f"),
        "Máximo": st.column_config.NumberColumn(format="%.2f"),
    }

    with tab_forecast:
        future = result.future[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={
                "ds": "Período", "yhat": "Previsto",
                "yhat_lower": "Mínimo", "yhat_upper": "Máximo",
            }
        )
        st.dataframe(
            future, use_container_width=True, hide_index=True,
            column_config=column_config,
        )
        st.download_button(
            "Baixar projeção (CSV)",
            future.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name=f"forecast_{value_col}.csv",
            mime="text/csv",
        )

    with tab_full:
        full = result.frame.rename(
            columns={
                "ds": "Período", "y": "Real", "yhat": "Previsto",
                "yhat_lower": "Mínimo", "yhat_upper": "Máximo", "tipo": "Tipo",
            }
        )
        st.dataframe(
            full, use_container_width=True, hide_index=True,
            column_config={**column_config, "Real": st.column_config.NumberColumn(format="%.2f")},
        )
        st.download_button(
            "Baixar série completa (CSV)",
            full.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name=f"serie_completa_{value_col}.csv",
            mime="text/csv",
        )

    with tab_diag:
        metrics = result.metrics
        cols = st.columns(4)
        cols[0].metric("MAPE", f"{_fmt(metrics['mape'], 2)}%")
        cols[1].metric("MAE", _fmt(metrics["mae"]))
        cols[2].metric("RMSE", _fmt(metrics["rmse"]))
        cols[3].metric("R²", _fmt(metrics["r2"], 3))
        st.plotly_chart(residuals_chart(result), use_container_width=True)
        st.caption(
            f"Linhas lidas: {series.rows_in} · utilizadas: {series.rows_out} · "
            f"datas inválidas: {series.dropped_invalid_date} · "
            f"valores inválidos: {series.dropped_invalid_value} · "
            f"duplicadas agregadas: {series.duplicates_aggregated}"
        )


if __name__ == "__main__":
    main()
