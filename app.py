import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# Título do App
st.title("P&L - ARTEFACT")

# Subtítulo do App
st.subheader("Análise preditiva de P&L utilizando modelo de previsão Prophet")

# Upload do arquivo via Drag-and-Drop
uploaded_file = st.file_uploader("Arraste e solte a base de dados aqui", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Verifica o tipo de arquivo e lê os dados
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')): 
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado. Use CSV, XLS ou XLSX.")
            st.stop()

        # Mostra os dados carregados
        st.subheader("Dados Carregados (Apenas 10 Primeiras Linhas):")
        st.dataframe(data.head(10))

        # Seleção de colunas
        st.subheader("Configuração do Forecast")
        try:
            date_column = st.selectbox("Selecione a coluna de data:", data.columns)
            value_column = st.selectbox("Selecione a coluna de valores:", data.columns)

            if date_column and value_column:
                # Renomeia as colunas para o formato esperado pelo Prophet
                forecast_data = data[[date_column, value_column]].rename(
                    columns={date_column: "ds", value_column: "y"}
                )
                forecast_data["ds"] = pd.to_datetime(forecast_data["ds"], format='%d/%m/%Y')

                # Criação do modelo Prophet
                model = Prophet()
                model.fit(forecast_data)

                # Input do usuário para o número de meses de forecast
                forecast_months = st.slider(
                    "Selecione quantos meses à frente deseja prever:",
                    min_value=1,
                    max_value=12,
                    value=6,
                    step=1
                )

                # Geração do forecast
                future = model.make_future_dataframe(periods=forecast_months, freq="M")
                forecast = model.predict(future)

                # Adicionar coluna indicando se é histórico ou forecast
                forecast['type'] = forecast['ds'].apply(
                    lambda x: 'Histórico' if x <= pd.to_datetime('today') else 'Forecast'
                )

                # Exibição do resultado com a coluna extra
                st.subheader(f"Tabela do Forecast com Histórico e Forecast")
                st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "type"]])

                # Criação do gráfico
                fig = go.Figure()

                # Linha de dados históricos com marcadores
                fig.add_trace(go.Scatter(
                    x=forecast_data["ds"],
                    y=forecast_data["y"],
                    mode='lines+markers',  # Marcadores para o histórico
                    name="Histórico",
                    line=dict(color='blue', width=2)
                ))

                # Linha de previsão (yhat)
                fig.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    mode='lines',
                    name="Previsão (yhat)",
                    line=dict(color='green', width=2)
                ))

                # Linha superior (yhat_upper)
                fig.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_upper"],
                    mode='lines',
                    name="Limite Superior (yhat_upper)",
                    line=dict(color='orange', width=1, dash='dash')
                ))

                # Linha inferior (yhat_lower)
                fig.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_lower"],
                    mode='lines',
                    name="Limite Inferior (yhat_lower)",
                    line=dict(color='red', width=1, dash='dash')
                ))

                # Configuração do layout do gráfico
                fig.update_layout(
                    title=f"Forecast de {forecast_months} Meses",
                    xaxis_title="Data",
                    yaxis_title="Valor",
                    legend_title="Legenda",
                    template="plotly_white"
                )

                # Exibir o gráfico no Streamlit
                st.plotly_chart(fig)

            else:
                st.warning("Selecione as colunas de data e valores para continuar.")

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
