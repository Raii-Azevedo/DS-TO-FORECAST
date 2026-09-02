"""Configurações globais: identidade visual, limites do modelo e frequências suportadas."""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Identidade visual Artefact
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Brand:
    navy: str = "#12123B"
    navy_soft: str = "#1E1E5A"
    pink: str = "#FF5C8A"
    pink_soft: str = "#FFE3EB"
    violet: str = "#7B61FF"
    cyan: str = "#00C2CB"
    amber: str = "#FFB020"
    red: str = "#E5484D"
    green: str = "#2FA84F"
    ink: str = "#1A1A2E"
    muted: str = "#6B7280"
    line: str = "#E5E7EB"
    surface: str = "#FFFFFF"
    canvas: str = "#F7F8FC"
    font: str = "Roboto, 'Helvetica Neue', Arial, sans-serif"


BRAND = Brand()


# --------------------------------------------------------------------------- #
# Parâmetros do forecast
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ForecastSettings:
    #: mínimo absoluto de pontos que o Prophet aceita sem falhar na inicialização
    min_points: int = 2
    #: abaixo disso o Prophet fica instável — usamos fallback determinístico
    min_points_reliable: int = 6
    #: a partir daqui faz sentido habilitar sazonalidade anual
    min_points_yearly_seasonality: int = 24
    max_horizon: int = 36
    default_horizon: int = 6
    interval_width: float = 0.80
    seed: int = 42


SETTINGS = ForecastSettings()


#: rótulo legível -> alias de frequência do pandas
FREQUENCIES: dict[str, str] = {
    "Diária": "D",
    "Semanal": "W",
    "Mensal": "MS",
    "Trimestral": "QS",
    "Anual": "YS",
}

#: pistas de nome de coluna usadas na detecção automática
DATE_HINTS: tuple[str, ...] = (
    "data", "date", "dt", "mes", "mês", "month", "periodo", "período",
    "competencia", "competência", "ds", "ref",
)

VALUE_HINTS: tuple[str, ...] = (
    "valor", "value", "receita", "revenue", "custo", "cost", "margem",
    "margin", "gm", "ebitda", "total", "amount", "y", "vendas", "sales",
    "faturado", "faturamento", "preco", "preço", "price", "quantidade",
    "qtd", "qty", "volume",
)

#: pistas de identificador — colunas assim são numéricas mas não são métricas,
#: então são despriorizadas na sugestão automática de coluna de valor.
ID_HINTS: tuple[str, ...] = (
    "cd ", "cd_", "id ", "id_", "cod", "código", "codigo", "nr ", "nr_",
    "num ", "chave", "key", "matricula", "matrícula", "cnpj", "cpf",
)


@dataclass(frozen=True)
class AppInfo:
    title: str = "Dataset to Forecast"
    subtitle: str = "Projeção de séries temporais a partir do seu dataset"
    owner: str = "Artefact"
    max_preview_rows: int = 200
    supported_extensions: tuple[str, ...] = field(
        default=(".csv", ".xlsx", ".xls")
    )


APP = AppInfo()
