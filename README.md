# P&L Forecast

Aplicação Streamlit para previsão de séries financeiras (P&L) a partir de um arquivo
CSV ou Excel, usando **Prophet** com fallback determinístico.

Carregue a base, confirme as colunas de data e valor, escolha o horizonte e receba
a projeção com intervalo de confiança, métricas de qualidade e exportação em CSV.

---

## Instalação

```bash
python -m venv streamlitenv
source streamlitenv/bin/activate      # Windows: streamlitenv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.9+.

---

## Formato de entrada

Basta uma coluna de data e uma de valor — nome e ordem são livres, a detecção é
automática.

| Data       | Valor        |
|------------|--------------|
| 01/01/2025 | R$ 120.000,00|
| 01/02/2025 | 138.500,50   |
| 01/03/2025 | 141.200,75   |

A camada de saneamento aceita, sem intervenção manual:

- formato brasileiro (`1.234,56`) e americano (`1,234.56`);
- símbolos de moeda e percentual (`R$`, `€`, `%`);
- negativos contábeis entre parênteses — `(1.234,56)` vira `-1234.56`;
- tokens de vazio: `N/A`, `-`, `#DIV/0!`, `#REF!`, células em branco;
- datas em `DD/MM/AAAA` ou `AAAA-MM-DD`, com ou sem timezone;
- datas repetidas, consolidadas pelo critério escolhido (soma, média, último…);
- CSV com separador `,` ou `;` e encoding UTF-8 ou Latin-1.

Linhas irrecuperáveis são descartadas e reportadas na aba **Diagnóstico**.

---

## Arquitetura

```
app.py                 Interface Streamlit (apenas UI e orquestração)
src/
  config.py            Paleta Artefact, limites do modelo, frequências
  data_loader.py       Leitura CSV/Excel + detecção automática de colunas
  preprocessing.py     Coerção de tipos, limpeza, deduplicação, validação
  forecasting.py       Prophet em cascata + métricas
  charts.py            Gráficos Plotly com a paleta da marca
  theme.py             CSS e componentes visuais
tests/
  test_pipeline.py     26 testes do pipeline (pytest)
```

A separação isola a lógica de negócio da UI: `src/` não importa `streamlit`, o que
torna o pipeline testável fora do navegador e reaproveitável em um job batch.

---

## Robustez do modelo

O erro original em produção era a falha de inicialização do Stan:

```
Exception: normal_lpdf: Random variable is nan, but must be not nan!
Initialization between (-2, 2) failed after 1 attempts.
```

Ele ocorre quando a coluna `y` chega ao `model.fit()` com `NaN`, infinito ou
tipo texto — o código anterior renomeava as colunas e chamava `fit()` sem nenhuma
coerção numérica. Duas camadas resolvem isso:

**1. Saneamento (`preprocessing.py`)** — a série é convertida para `float64`,
valores não finitos são removidos, datas duplicadas agregadas e a série ordenada.
Uma verificação `np.isfinite` roda como última barreira antes do modelo. Problemas
estruturais (série vazia, coluna errada, pontos insuficientes) viram `ValidationError`
com mensagem em português, não um crash.

**2. Cascata de motores (`forecasting.py`)** — se um nível falha, o próximo assume:

| Nível | Motor | Quando entra |
|-------|-------|--------------|
| 1 | Prophet com a configuração escolhida | caso normal |
| 2 | Prophet simplificado (sem sazonalidade, growth linear) | o nível 1 não converge |
| 3 | Tendência linear (OLS) com banda empírica | série < 6 pontos, ou o nível 2 falha |

O usuário sempre recebe uma projeção, e o motor efetivamente usado aparece na tela.
Outras proteções: horizonte limitado a 36 períodos, frequência inferida dos dados
(em vez de fixada em mensal), MAPE que ignora zeros no denominador, e
`interval_width` explícito de 80%.

---

## Métricas

A aba **Diagnóstico** traz MAPE, MAE, RMSE e R² sobre o período histórico, mais o
gráfico de resíduos. Leitura sugerida do MAPE: abaixo de 10% excelente, até 20%
boa, até 35% razoável — acima disso, revise a base antes de usar a projeção.

---

## Testes

```bash
pytest -q
```

Cobrem conversão numérica, limpeza, inferência de frequência e forecast em casos
adversos: séries com NaN, valores em texto, constantes, com zeros, com dois pontos,
com datas irregulares e com horizonte fora do limite.

---

## Autora

Raíssa Azevedo · Artefact
