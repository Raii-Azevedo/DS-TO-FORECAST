"""Leitura de arquivos e detecção automática de colunas.

Aceita CSV (com separador e encoding descobertos automaticamente) e Excel.
Nunca levanta exceção crua para a UI: erros viram ``DataLoadError`` com
mensagem em português explicando o que fazer.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .config import APP, DATE_HINTS, VALUE_HINTS


class DataLoadError(Exception):
    """Erro previsível de leitura de arquivo, já com mensagem amigável."""


_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")
_CSV_SEPARATORS = (None, ";", ",", "\t", "|")  # None => sniffing do pandas


def _read_csv(raw: bytes) -> pd.DataFrame:
    """Tenta combinações de encoding/separador até obter mais de uma coluna."""
    last_error: Exception | None = None
    best: pd.DataFrame | None = None

    for encoding in _CSV_ENCODINGS:
        for sep in _CSV_SEPARATORS:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=encoding,
                    sep=sep,
                    engine="python" if sep is None else "c",
                )
            except Exception as exc:  # noqa: BLE001 - tentativa exploratória
                last_error = exc
                continue

            if df.shape[1] > 1:
                return df
            if best is None:
                best = df

    if best is not None:
        return best

    raise DataLoadError(
        "Não foi possível ler o CSV. Verifique o separador (`,` ou `;`) e o "
        f"encoding do arquivo. Detalhe técnico: {last_error}"
    )


def _read_excel(raw: bytes, sheet_name: str | int = 0) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name)
    except ImportError as exc:  # openpyxl ausente
        raise DataLoadError(
            "Leitura de Excel indisponível: instale `openpyxl` "
            "(`pip install -r requirements.txt`)."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Não foi possível ler a planilha: {exc}") from exc


def excel_sheet_names(raw: bytes) -> list[str]:
    """Lista as abas de um arquivo Excel (vazio se não for legível)."""
    try:
        return list(pd.ExcelFile(io.BytesIO(raw)).sheet_names)
    except Exception:  # noqa: BLE001
        return []


def load_dataframe(
    raw: bytes,
    filename: str,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Carrega bytes de um upload em um ``DataFrame`` limpo.

    Args:
        raw: conteúdo binário do arquivo.
        filename: nome original, usado para inferir o formato.
        sheet_name: aba do Excel (ignorado para CSV).

    Raises:
        DataLoadError: formato não suportado ou arquivo ilegível/vazio.
    """
    suffix = Path(filename).suffix.lower()

    if suffix not in APP.supported_extensions:
        raise DataLoadError(
            f"Formato `{suffix or 'desconhecido'}` não suportado. "
            f"Use {', '.join(APP.supported_extensions)}."
        )

    df = _read_csv(raw) if suffix == ".csv" else _read_excel(raw, sheet_name)

    # Normaliza cabeçalhos e descarta linhas/colunas totalmente vazias.
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if df.empty:
        raise DataLoadError("O arquivo foi lido, mas não contém nenhuma linha de dados.")
    if df.shape[1] < 2:
        raise DataLoadError(
            "O arquivo precisa de pelo menos duas colunas: uma de data e uma de valores."
        )

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Perfilamento e detecção automática de colunas
# --------------------------------------------------------------------------- #
#: mínimo de linhas convertidas com sucesso para a coluna ser considerada usável
USABLE_THRESHOLD = 0.60

#: linhas amostradas no perfilamento — mantém a detecção rápida em bases grandes
_PROFILE_SAMPLE = 3_000


def _score(column: str, hints: tuple[str, ...]) -> int:
    """Pontua o nome da coluna contra a lista de pistas (0 = nenhuma pista)."""
    name = column.strip().lower()
    for position, hint in enumerate(hints):
        if name == hint:
            return 100 - position
        if hint in name:
            return 50 - position
    return 0


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mede, por coluna, o quanto ela serve como data e como valor.

    A checagem é feita sobre uma amostra e importa mais que o nome: uma coluna
    chamada "Data Prevista" que na verdade guarda ``True``/``False`` recebe
    ``date_ratio`` zero e deixa de ser sugerida.

    Returns:
        DataFrame indexado pelo nome da coluna, com ``date_ratio``,
        ``value_ratio`` (0..1) e as pontuações de nome.
    """
    from .preprocessing import to_datetime, to_numeric  # import tardio: evita ciclo

    sample = df.sample(min(len(df), _PROFILE_SAMPLE), random_state=0) if len(df) else df
    rows = []

    for column in df.columns:
        values = sample[column]
        non_null = values.notna()
        base = int(non_null.sum())

        if base == 0 or pd.api.types.is_bool_dtype(values):
            # Coluna vazia ou booleana (checkbox) não serve para nada aqui.
            date_ratio = value_ratio = 0.0
        else:
            date_ratio = float(to_datetime(values).notna().sum()) / base
            value_ratio = float(to_numeric(values).notna().sum()) / base

        rows.append(
            {
                "column": column,
                "date_ratio": round(date_ratio, 4),
                "value_ratio": round(value_ratio, 4),
                "date_name_score": _score(column, DATE_HINTS),
                "value_name_score": _score(column, VALUE_HINTS),
            }
        )

    return pd.DataFrame(rows).set_index("column")


def usable_date_columns(profile: pd.DataFrame) -> list[str]:
    """Colunas que realmente convertem em data, da melhor para a pior."""
    usable = profile[profile["date_ratio"] >= USABLE_THRESHOLD]
    ordered = usable.sort_values(
        ["date_name_score", "date_ratio"], ascending=[False, False]
    )
    return list(ordered.index)


def usable_value_columns(profile: pd.DataFrame, exclude: str | None = None) -> list[str]:
    """Colunas que realmente convertem em número, da melhor para a pior."""
    usable = profile[profile["value_ratio"] >= USABLE_THRESHOLD]
    if exclude is not None:
        usable = usable.drop(index=exclude, errors="ignore")
    ordered = usable.sort_values(
        ["value_name_score", "value_ratio"], ascending=[False, False]
    )
    return list(ordered.index)


def suggest_date_column(df: pd.DataFrame, profile: pd.DataFrame | None = None) -> str:
    """Melhor candidata a coluna de data; cai na primeira coluna se nenhuma servir."""
    profile = profile_columns(df) if profile is None else profile
    candidates = usable_date_columns(profile)
    if candidates:
        return candidates[0]
    return str(profile["date_ratio"].idxmax()) if len(profile) else str(df.columns[0])


def suggest_value_column(
    df: pd.DataFrame,
    exclude: str | None = None,
    profile: pd.DataFrame | None = None,
) -> str:
    """Melhor candidata a coluna de valores; cai na mais numérica se nenhuma servir."""
    profile = profile_columns(df) if profile is None else profile
    candidates = usable_value_columns(profile, exclude=exclude)
    if candidates:
        return candidates[0]

    fallback = profile.drop(index=exclude, errors="ignore") if exclude else profile
    if len(fallback):
        return str(fallback["value_ratio"].idxmax())
    return str(df.columns[0])
