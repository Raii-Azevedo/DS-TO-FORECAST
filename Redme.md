P&L Forecasting Tool

Este documento descreve e documenta o funcionamento do P&L Forecasting Tool, um aplicativo interativo desenvolvido em Python usando o framework Streamlit e o modelo de previsão Prophet. Ele permite carregar dados históricos de receita ou valores financeiros e realizar previsões futuras baseadas nas tendências do histórico.

Objetivo

Fornecer um método intuitivo e eficiente para:

Visualizar dados históricos de P&L (Profit and Loss).

Prever valores futuros (Forecast) utilizando o modelo Prophet.

Exibir resultados em gráficos interativos e tabelas estilizadas.

Requisitos de Entrada

O programa exige um arquivo de entrada contendo:

Coluna de data: Com formato DD/MM/AAAA.

Coluna de valores: Contendo valores numéricos relacionados ao P&L.

O arquivo pode estar nos formatos .csv, .xls ou .xlsx.

Instalação

Certifique-se de instalar as dependências antes de executar o programa:

pip install streamlit prophet plotly pandas scikit-learn

Fluxo do Código

1. Upload e Processamento de Arquivo

O aplicativo permite ao usuário fazer upload de um arquivo de dados e seleciona as colunas para análise:

uploaded_file = st.file_uploader("Arraste e solte a base de dados aqui", type=["csv", "xlsx", "xls"])

Tratamento do arquivo: Verifica o tipo de arquivo (CSV ou Excel) e lê os dados.

Exibição de dados: Mostra as 10 primeiras linhas para visualização inicial:

st.dataframe(data.head(10))

2. Configuração de Forecast

O usuário seleciona:

A coluna de datas (campo ds).

A coluna de valores (campo y).

Os dados são renomeados para o formato exigido pelo Prophet:

forecast_data = data[[date_column, value_column]].rename(columns={date_column: "ds", value_column: "y"})
forecast_data["ds"] = pd.to_datetime(forecast_data["ds"], format="%d/%m/%Y")

3. Modelo Prophet

O modelo Prophet é criado e ajustado com os dados históricos:

model = Prophet()
model.fit(forecast_data)

Em seguida, gera previsões futuras:

future = model.make_future_dataframe(periods=forecast_months, freq="M")
forecast = model.predict(future)

4. Separação de Histórico e Forecast

O código distingue entre dados históricos e previsões:

forecast['type'] = forecast['ds'].apply(lambda x: 'Histórico' if x <= last_date else 'Forecast')

5. Gráficos

Histórico e Forecast

Gráfico de linhas interativo com Plotly, exibindo:

Dados históricos.

Previsões (yhat).

Limites superior (yhat_upper) e inferior (yhat_lower).

fig.add_trace(go.Scatter(x=forecast_data["ds"], y=forecast_data["y"], mode='lines', name="Histórico", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name="Previsão (yhat)", line=dict(color='green')))

Gráfico de Barras

Barras agrupadas mostrando:

Valores previstos (yhat).

Limites inferior (yhat_lower) e superior (yhat_upper).

forecast_fig.add_trace(go.Bar(x=forecast_only["ds"], y=forecast_only["yhat"], name="Previsão (Yhat)", marker_color='lightgreen'))

6. Tabelas Estilizadas

Exibe uma tabela com os valores previstos, estilizando diferentes colunas com cores distintas:

def color_forecast(val):
    if val.name == "Previsão (Yhat)":
        return ['background-color: lightgreen' for _ in val]
    elif val.name == "Limite Inferior (Yhat Lower)":
        return ['background-color: lightcoral' for _ in val]
    elif val.name == "Limite Superior (Yhat Upper)":
        return ['background-color: lightblue' for _ in val]
    return ['' for _ in val]

Resultados

1. Gráfico Interativo

Dados históricos e previsões são representados em um gráfico de linha.

Limites superior e inferior são exibidos como linhas tracejadas para indicar variação.

2. Tabela de Forecast

Tabela detalhada com previsões futuras, destacando os valores principais.

Personalizações Futuras

Adição de sazonalidade: Incorporar fatores sazonais específicos do negócio.

Suporte para outros idiomas: Melhorar a internacionalização do aplicativo.

Análise de erro: Incluir mais métricas de erro além do MAPE.

Uso

Execute o aplicativo com o comando:

streamlit run app.py

Substitua app.py pelo nome do arquivo do script Python.

Conclusão

Este aplicativo fornece uma solução prática e escalável para prever tendências financeiras, utilizando técnicas modernas de aprendizado de máquina e uma interface amigável. Personalize conforme necessário para atender às suas necessidades de análise.


