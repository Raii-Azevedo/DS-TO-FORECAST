import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Título do App
st.title("P&L - ARTEFACT")

# Subtítulo do App
st.subheader("Análise preditiva de P&L utilizando modelo de previsão Prophet")

# Upload do arquivo via Drag-and-Drop
uploaded_file = st.file_uploader("Arraste e solte a base de dados aqui", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        # Carregar dados
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Exibição inicial dos dados
        st.subheader("Dados Carregados:")
        st.dataframe(data.head())

        # Configuração de colunas
        st.subheader("Configuração do Forecast")
        date_column = st.selectbox("Selecione a coluna de data:", data.columns)
        value_column = st.selectbox("Selecione a coluna de valores:", data.columns)

        if date_column and value_column:
            # Renomear colunas para o Prophet
            data = data[[date_column, value_column]].rename(columns={date_column: "ds", value_column: "y"})
            data["ds"] = pd.to_datetime(data["ds"])

            # Exibir os dados históricos completos
            st.write("Dados históricos:")
            st.dataframe(data)

            # Criar modelo Prophet
            model = Prophet()
            model.fit(data)

            # Determinar a última data registrada nos dados
            last_date = data["ds"].max()

            # Input do usuário para o número de meses de previsão
            st.subheader("Configuração do Período de Forecast")
            forecast_months = st.slider(
                "Quantos meses à frente você deseja prever?",
                min_value=1,
                max_value=12,
                value=3,
                step=1
            )

            # Geração do DataFrame futuro com base no input do usuário
            future = model.make_future_dataframe(periods=forecast_months, freq="MS")  # Primeiros dias dos meses
            future = future[future["ds"] > last_date]  # Incluir apenas datas após o último mês registrado
            forecast = model.predict(future)

            # Adicionar colunas de tipo para identificar histórico e previsão
            forecast["type"] = "Forecast"

            # Ajustar formato de datas para manter apenas o primeiro dia do mês
            forecast["ds"] = forecast["ds"].dt.to_period("M").dt.to_timestamp()

            # Exibir tabela de forecast
            st.subheader("Tabela de Previsão")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "type"]])

            # Criar gráfico
            fig = go.Figure()

            # Dados históricos (apenas linha azul para histórico real)
            fig.add_trace(go.Scatter(
                x=data["ds"], y=data["y"], mode='lines+markers', name="Histórico", line=dict(color='blue', width=2)
            ))

            # Previsão (yhat)
            fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat"],
                mode='lines',
                name="Previsão",
                line=dict(color='green', width=2)
            ))

            # Limites de confiança
            fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat_upper"],
                mode='lines',
                name="Limite Superior",
                line=dict(color='orange', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat_lower"],
                mode='lines',
                name="Limite Inferior",
                line=dict(color='red', dash='dash')
            ))

            # Layout do gráfico
            fig.update_layout(
                title="Previsão com Prophet",
                xaxis_title="Data",
                yaxis_title="Valores",
                legend_title="Legenda",
                template="plotly_white"
            )

            # Exibir gráfico
            st.plotly_chart(fig)

            # Comparação de acuracidade (opcional)
            st.subheader("Métricas de Acuracidade:")
            mae = mean_absolute_error(data["y"], model.predict(data)["yhat"])
            mse = mean_squared_error(data["y"], model.predict(data)["yhat"])
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(data["y"], model.predict(data)["yhat"])

            st.write(f"Erro Absoluto Médio (MAE): {mae:.2f}")
            st.write(f"Erro Quadrático Médio (MSE): {mse:.2f}")
            st.write(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
            st.write(f"Erro Percentual Absoluto Médio (MAPE): {mape * 100:.2f}%")

    except Exception as e:
        st.error(f"Erro: {e}")
