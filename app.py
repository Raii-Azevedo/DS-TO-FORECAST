import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error

# Título do App
st.title("P&L - ARTEFACT")

# Subtítulo do App
st.subheader("Análise preditiva de P&L utilizando modelo de previsão Prophet")

# Upload do arquivo via Drag-and-Drop
uploaded_file = st.file_uploader("Arraste e solte a base de dados aqui", type=["csv", "xlsx", "xls"])

if uploaded_file:
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
        date_column = st.selectbox("Selecione a coluna de data:", data.columns)
        value_column = st.selectbox("Selecione a coluna de valores:", data.columns)

        if date_column and value_column:
            forecast_data = data[[date_column, value_column]].rename(
                columns={date_column: "ds", value_column: "y"}
            )
            forecast_data["ds"] = pd.to_datetime(forecast_data["ds"], format="%d/%m/%Y")

            # Criação do modelo Prophet
            model = Prophet()
            model.fit(forecast_data)

            # Identificar o último mês com dados históricos
            last_date = forecast_data['ds'].max()

            # Input do usuário para o número de meses de forecast
            forecast_months = st.slider(
                "Selecione quantos meses à frente deseja prever:",
                min_value=1,
                max_value=12,
                value=6,
                step=1
            )

            # Geração do forecast a partir do mês seguinte ao último mês registrado
            future = model.make_future_dataframe(periods=forecast_months, freq="M")
            forecast = model.predict(future)

            # Combina os dados históricos e previsões
            forecast = pd.merge(
                future,
                forecast_data[['ds', 'y']],
                on='ds',
                how='left'
            )

            # Ajusta os valores de histórico e previsão
            forecast['y'] = forecast['y'].fillna(forecast['yhat'])
            forecast['type'] = forecast['ds'].apply(
                lambda x: 'Histórico' if x <= last_date else 'Forecast'
            )

            # Cálculo do MAPE (Mean Absolute Percentage Error) para os dados históricos
            historical_values = forecast_data.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
            mape = mean_absolute_percentage_error(historical_values['y'], historical_values['yhat'])
            st.subheader(f"MAPE (Mean Absolute Percentage Error): {mape * 100:.2f}%")

            # Exibição do resultado com a coluna extra
            st.subheader("Tabela do Forecast com Histórico e Forecast")
            st.dataframe(forecast[["ds", "y", "yhat", "yhat_lower", "yhat_upper", "type"]])

            # Criação do gráfico
            fig = go.Figure()

            # Linha de dados históricos
            fig.add_trace(go.Scatter(
                x=forecast[forecast['type'] == 'Histórico']["ds"],
                y=forecast[forecast['type'] == 'Histórico']["y"],
                mode='lines',
                name="Histórico",
                line=dict(color='blue', width=1)
            ))

            # Linha de previsão (yhat)
            fig.add_trace(go.Scatter(
                x=forecast[forecast['type'] == 'Forecast']["ds"],
                y=forecast[forecast['type'] == 'Forecast']["y"],
                mode='lines',
                name="Previsão (yhat)",
                line=dict(color='green', width=2)
            ))

            # Configuração do layout do gráfico
            fig.update_layout(
                title=f"Forecast de {forecast_months} Meses",
                xaxis_title="Data",
                yaxis_title="Valor",
                legend_title="Legenda",
                template="plotly_white"
            )

            st.plotly_chart(fig)

        else:
            st.warning("Selecione as colunas de data e valores para continuar.")

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
