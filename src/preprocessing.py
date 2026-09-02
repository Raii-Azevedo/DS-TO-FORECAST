"""Saneamento da série temporal antes do Prophet.

Esta é a camada que previne o erro clássico do Stan:

    Exception: normal_lpdf: Random variable is nan, but must be not nan!

Ele acontece quando a coluna `y` chega ao `model.fit()` com NaN, infinito ou
com tipo texto. Aqui a série é convertida, limpa, deduplicada e ordenada, e
qualquer problema estrutural vira um `ValidationError` legível — nunca um
crash do modelo.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SETTINGS


class ValidationError(Exception):
    """Série inadequada para forecast, com explicação para o usuário."""


@dataclass
class CleanSeries:
    """Série pronta para o Prophet, acompanhada do relatório de limpeza."""

    df: pd.DataFrame                     # colunas: ds (datetime64), y (float)
    freq: str                            # alias pandas inferido (ex.: "MS")
    freq_label: str                      # rótulo legível (ex.: "Mensal")
    rows_in: int = 0
    rows_out: int = 0
    dropped_invalid_date: int = 0
    dropped_invalid_value: int = 0
    duplicates_aggregated: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def dropped_total(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def start(self) -> pd.Timestamp:
        return self.df["ds"].min()

    @property
    def end(self) -> pd.Timestamp:
        return self.df["ds"].max()


# --------------------------------------------------------------------------- #
# Conversão numérica tolerante
# --------------------------------------------------------------------------- #
_CURRENCY_NOISE = re.compile(r"[R$€£¥%\s ]", re.IGNORECASE)

#: forma de um número completo — a string precisa casar por inteiro. Letras não
#: são removidas: se fossem, códigos como "BU02" ou "BR014074" virariam números.
_NUMERIC_SHAPE = re.compile(r"^[+-]?[0-9][0-9.,]*(?:[eE][+-]?[0-9]+)?$")

_NULL_TOKENS = {
    "", "-", "--", "n/a", "na", "nan", "none", "null", "nd", "n.d.",
    "#n/a", "#div/0!", "#value!", "#ref!", "sem dados", "s/d",
}


def _parse_number(token: object) -> float:
    """Converte um valor solto em float, entendendo formato BR e contábil.

    Trata: "R$ 1.234,56", "(1.234,56)" (negativo contábil), "1,234.56",
    "12%", separador de milhar, espaço não-quebrável e tokens de vazio.
    Retorna ``np.nan`` quando não há número reconhecível.
    """
    if token is None:
        return np.nan
    if isinstance(token, (int, float, np.integer, np.floating)):
        value = float(token)
        return value if np.isfinite(value) else np.nan

    text = str(token).strip()
    if text.lower() in _NULL_TOKENS:
        return np.nan

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    text = _CURRENCY_NOISE.sub("", text)
    if not _NUMERIC_SHAPE.match(text):
        # Sobrou letra ou símbolo: é um código ("BU02", "BR014074"), não um número.
        return np.nan

    has_comma, has_dot = "," in text, "." in text
    if has_comma and has_dot:
        # O separador decimal é o que aparece por último.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        # "1,5" é decimal; "1,234" com 3 dígitos finais é milhar.
        integer, _, decimals = text.rpartition(",")
        text = (
            text.replace(",", "")
            if len(decimals) == 3 and integer.replace("-", "").isdigit()
            else text.replace(",", ".")
        )

    try:
        value = float(text)
    except ValueError:
        return np.nan
    if not np.isfinite(value):
        return np.nan
    return -abs(value) if negative else value


def to_numeric(series: pd.Series) -> pd.Series:
    """Coerção numérica de uma coluna inteira, sem levantar exceção.

    Datas e booleanos são rejeitados de propósito: `pd.to_numeric` converteria
    um datetime em nanossegundos desde 1970, fazendo uma coluna de data passar
    por métrica na detecção automática.
    """
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_bool_dtype(series):
        return pd.Series(np.nan, index=series.index, dtype="float64")

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        fast = pd.to_numeric(series, errors="coerce")
        # Só paga o custo do parser textual se a via rápida perdeu dados.
        numeric = fast if fast.notna().all() else series.map(_parse_number)
    return pd.to_numeric(numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).astype("float64")


def to_datetime(series: pd.Series) -> pd.Series:
    """Conversão de datas tolerante, com preferência por dia/mês/ano.

    Colunas puramente numéricas só são aceitas no formato compacto AAAAMMDD ou
    AAAAMM. Sem essa trava, o pandas leria um identificador como `324850` como
    epoch em nanossegundos e produziria uma data de 1970 — um ID de pedido
    viraria silenciosamente uma coluna de datas.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    if pd.api.types.is_bool_dtype(series):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if pd.api.types.is_numeric_dtype(series):
        numbers = pd.to_numeric(series, errors="coerce")
        compact = numbers.dropna()
        if compact.empty:
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

        as_int = compact.astype("int64").astype(str)
        if as_int.str.len().eq(8).all():
            return pd.to_datetime(numbers, format="%Y%m%d", errors="coerce")
        if as_int.str.len().eq(6).all():
            return pd.to_datetime(numbers, format="%Y%m", errors="coerce")
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    with warnings.catch_warnings():
        # O pandas avisa quando cai no dateutil por não inferir um formato único.
        # É o comportamento desejado aqui: a base pode misturar formatos.
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if parsed.notna().mean() < 0.5:
            alternative = pd.to_datetime(series, errors="coerce", dayfirst=False)
            if alternative.notna().mean() > parsed.notna().mean():
                parsed = alternative

    # Remove timezone: o Prophet só aceita datetimes naive.
    if getattr(parsed.dtype, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


# --------------------------------------------------------------------------- #
# Inferência de frequência
# --------------------------------------------------------------------------- #
_FREQ_LABELS = {"D": "Diária", "W": "Semanal", "MS": "Mensal", "QS": "Trimestral", "YS": "Anual"}


def infer_frequency(dates: pd.Series) -> tuple[str, str]:
    """Descobre a granularidade da série a partir do intervalo mediano."""
    unique = pd.Series(pd.to_datetime(dates)).dropna().drop_duplicates().sort_values()
    if len(unique) < 2:
        return "MS", "Mensal"

    if len(unique) >= 3:
        # `infer_freq` exige 3+ datas e lança exceção em séries irregulares.
        try:
            inferred = pd.infer_freq(pd.DatetimeIndex(unique))
        except (ValueError, TypeError):
            inferred = None
        if inferred:
            head = inferred.split("-")[0].upper()
            for alias, label in _FREQ_LABELS.items():
                if head.startswith(alias) or head.startswith(alias[0]):
                    return alias, label

    days = float(unique.diff().dropna().dt.days.median() or 30)
    if days <= 2:
        return "D", "Diária"
    if days <= 10:
        return "W", "Semanal"
    if days <= 45:
        return "MS", "Mensal"
    if days <= 130:
        return "QS", "Trimestral"
    return "YS", "Anual"


# --------------------------------------------------------------------------- #
# Pipeline principal
# --------------------------------------------------------------------------- #
def prepare_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    aggregation: str = "sum",
    freq_override: str | None = None,
) -> CleanSeries:
    """Transforma duas colunas cruas em uma série válida para o Prophet.

    Passos: coerção de tipos -> descarte de datas/valores inválidos ->
    agregação de datas duplicadas -> ordenação -> validação final.

    Raises:
        ValidationError: coluna inexistente, série vazia, curta demais ou constante.
    """
    for column in (date_column, value_column):
        if column not in df.columns:
            raise ValidationError(f"A coluna `{column}` não existe no arquivo.")
    if date_column == value_column:
        raise ValidationError(
            "As colunas de data e de valor precisam ser diferentes."
        )

    rows_in = len(df)
    work = pd.DataFrame(
        {
            "ds": to_datetime(df[date_column]),
            "y": to_numeric(df[value_column]),
        }
    )

    invalid_dates = int(work["ds"].isna().sum())
    work = work[work["ds"].notna()]

    invalid_values = int(work["y"].isna().sum())
    work = work[work["y"].notna()]

    if work.empty:
        raise ValidationError(
            f"Nenhuma linha válida sobrou após a limpeza: `{date_column}` não "
            f"produziu datas reconhecíveis ou `{value_column}` não contém números. "
            "Confirme se selecionou as colunas certas."
        )

    duplicates = int(work["ds"].duplicated().sum())
    if duplicates:
        agg = aggregation if aggregation in {"sum", "mean", "median", "last", "max", "min"} else "sum"
        work = work.groupby("ds", as_index=False)["y"].agg(agg)

    work = work.sort_values("ds").reset_index(drop=True)
    work["y"] = work["y"].astype("float64")

    # Rede de segurança: garante que nada não-finito chegue ao Stan.
    work = work[np.isfinite(work["y"].to_numpy())]

    if len(work) < SETTINGS.min_points:
        raise ValidationError(
            f"São necessários pelo menos {SETTINGS.min_points} pontos válidos "
            f"para gerar um forecast — o arquivo forneceu {len(work)}."
        )

    freq, freq_label = infer_frequency(work["ds"])
    if freq_override:
        freq = freq_override
        freq_label = _FREQ_LABELS.get(freq_override, freq_override)

    warnings: list[str] = []
    if invalid_dates:
        warnings.append(f"{invalid_dates} linha(s) descartada(s) por data inválida.")
    if invalid_values:
        warnings.append(f"{invalid_values} linha(s) descartada(s) por valor não numérico ou vazio.")
    if duplicates:
        warnings.append(f"{duplicates} data(s) duplicada(s) agregada(s) por `{aggregation}`.")
    if len(work) < SETTINGS.min_points_reliable:
        warnings.append(
            f"Série curta ({len(work)} pontos). O intervalo de confiança será largo "
            "e a projeção, pouco confiável."
        )
    if float(work["y"].std(ddof=0)) == 0.0:
        warnings.append("A série é constante — a projeção repetirá o mesmo valor.")

    return CleanSeries(
        df=work,
        freq=freq,
        freq_label=freq_label,
        rows_in=rows_in,
        rows_out=len(work),
        dropped_invalid_date=invalid_dates,
        dropped_invalid_value=invalid_values,
        duplicates_aggregated=duplicates,
        warnings=warnings,
    )
