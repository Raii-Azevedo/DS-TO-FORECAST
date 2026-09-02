# Dataset to Forecast

Aplicação Streamlit que transforma qualquer dataset em uma projeção de série
temporal, usando **Prophet** com fallback determinístico.

Carregue um CSV ou Excel, confirme as colunas de data e valor, escolha o horizonte
e receba a projeção com intervalo de confiança, métricas de qualidade e exportação
em CSV.

O domínio dos dados é indiferente — a única exigência é uma coluna de data e uma de
valor numérico. Serve igualmente para vendas, volume de pedidos, headcount, consumo,
tempo de resposta, tráfego ou qualquer outra métrica com histórico.

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

## Detecção de colunas

Bases de BI costumam ter dezenas de colunas, e a maioria não serve para forecast.
Cada coluna é perfilada sobre uma amostra: mede-se quantas linhas convertem
efetivamente em data e quantas em número. Só as que passam de 60% são oferecidas
no seletor, com o percentual ao lado do nome.

Isso descarta armadilhas comuns de extração:

- colunas de **checkbox** (`True`/`False`), que não são datas;
- **identificadores inteiros** — sem a trava, `324850` viraria uma data de 1970,
  interpretado como epoch em nanossegundos;
- **códigos alfanuméricos** como `BU02` ou `BR014074`, que não podem virar 2 e
  14074 ao se removerem as letras;
- colunas de **data lidas como métrica** via nanossegundos.

Colunas cujo nome sugere identificador (`Cd `, `Id `, `Cód`, `Nr `) continuam
disponíveis, mas são despriorizadas na sugestão automática. Se nenhuma coluna
servir, a aplicação mostra o diagnóstico completo em vez de uma mensagem genérica.

---

## Nota sobre CSS no Streamlit

O `st.markdown` passa por um parser CommonMark antes de renderizar HTML, e duas
armadilhas fazem a folha de estilo aparecer como texto na tela: linhas indentadas
com 4+ espaços viram bloco de código, e uma linha em branco encerra o bloco HTML.
Por isso `theme.py` monta o CSS sem indentação e sem linhas em branco, e importa a
fonte com `@import` dentro do próprio `<style>`. Há um teste que protege isso.

O `requirements.txt` deve ficar em **UTF-8**. `pip freeze > requirements.txt` no
PowerShell grava em UTF-16, e o pip só consegue ler esse encoding quando o BOM
está presente — sem ele, o deploy falha antes de o app subir.

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
