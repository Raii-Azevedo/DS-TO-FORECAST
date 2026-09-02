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
# Detecção automática de colunas
# --------------------------------------------------------------------------- #
def _score(column: str, hints: tuple[str, ...]) -> int:
    name = column.strip().lower()
    for position, hint in enumerate(hints):
        if name == hint:
            return 100 - position
        if hint in name:
            return 50 - position
    return 0


def suggest_date_column(df: pd.DataFrame) -> str:
    """Melhor candidata a coluna de data: nome sugestivo ou maior taxa de parse."""
    by_name = sorted(df.columns, key=lambda c: _score(c, DATE_HINTS), reverse=True)
    if _score(by_name[0], DATE_HINTS) > 0:
        return by_name[0]

    best, best_ratio = df.columns[0], -1.0
    for column in df.columns:
        parsed = pd.to_datetime(df[column], errors="coerce", dayfirst=True)
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best, best_ratio = column, ratio
    return best


def suggest_value_column(df: pd.DataFrame, exclude: str | None = None) -> str:
    """Melhor candidata a coluna de valores: nome sugestivo ou coluna numérica."""
    candidates = [c for c in df.columns if c != exclude] or list(df.columns)

    by_name = sorted(candidates, key=lambda c: _score(c, VALUE_HINTS), reverse=True)
    if _score(by_name[0], VALUE_HINTS) > 0:
        return by_name[0]

    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else candidates[0]
