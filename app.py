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
        # Carregamento do arquivo
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado. Use CSV, XLS ou XLSX.")
            st.stop()

        # Exibição inicial dos dados
        st.subheader("Dados Carregados (Apenas 10 Primeiras Linhas):")
        st.dataframe(data.head(10))

        # Seleção de colunas
        st.subheader("Configuração do Forecast")
        date_column = st.selectbox("Selecione a coluna de data:", data.columns)
        value_column = st.selectbox("Selecione a coluna de valores:", data.columns)

        if date_column and value_column:
            # Prepara os dados
            forecast_data = data[[date_column, value_column]].rename(
                columns={date_column: "ds", value_column: "y"}
            )
            forecast_data["ds"] = pd.to_datetime(forecast_data["ds"], format="%d/%m/%Y")

            # Criação do modelo Prophet
            model = Prophet()
            model.fit(forecast_data)

            # Última data com valor
            last_date = forecast_data['ds'].max()

            # Configuração do forecast
            forecast_months = st.slider(
                "Selecione quantos meses à frente deseja prever:",
                min_value=1,
                max_value=12,
                value=6,
                step=1
            )

            # Gerar dados futuros
            future = model.make_future_dataframe(periods=forecast_months, freq="M")
            forecast = model.predict(future)

            # Combinação de dados históricos e previsão
            forecast['y'] = forecast['ds'].map(
                dict(zip(forecast_data['ds'], forecast_data['y']))
            )
            forecast['type'] = forecast['ds'].apply(
                lambda x: 'Histórico' if x <= last_date else 'Forecast'
            )
            forecast['y'] = forecast.apply(
                lambda row: row['y'] if row['type'] == 'Histórico' else row['yhat'], axis=1
            )

            # Cálculo do MAPE para os dados históricos
            historical_values = forecast_data.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
            mape = mean_absolute_percentage_error(historical_values['y'], historical_values['yhat'])
            st.subheader(f"MAPE (Mean Absolute Percentage Error): {mape * 100:.2f}%")

            # Exibição da tabela completa
            st.subheader("Tabela do Forecast com Histórico e Forecast")
            st.dataframe(forecast[["ds", "y", "yhat", "yhat_lower", "yhat_upper", "type"]])

            # Gráfico principal com Upper e Lower
            fig = go.Figure()

            # Linha de dados históricos
            fig.add_trace(go.Scatter(
                x=forecast[forecast['type'] == 'Histórico']["ds"],
                y=forecast[forecast['type'] == 'Histórico']["y"],
                mode='lines',
                name="Histórico",
                line=dict(color='blue', width=2)
            ))

            # Linha de previsão (yhat)
            fig.add_trace(go.Scatter(
                x=forecast[forecast['type'] == 'Forecast']["ds"],
                y=forecast[forecast['type'] == 'Forecast']["yhat"],
                mode='lines+markers',
                name="Previsão (yhat)",
                line=dict(color='green', width=2)
            ))

            # Linha superior (yhat_upper)
            fig.add_trace(go.Scatter(
                x=forecast[forecast['type'] == 'Forecast']["ds"],
                y=forecast[forecast['type'] == 'Forecast']["yhat_upper"],
                mode='lines',
                name="Limite Superior (yhat_upper)",
                line=dict(color='orange', width=1, dash='dash')
            ))

            # Linha inferior (yhat_lower)
            fig.add_trace(go.Scatter(
                x=forecast[forecast['type'] == 'Forecast']["ds"],
                y=forecast[forecast['type'] == 'Forecast']["yhat_lower"],
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

            st.plotly_chart(fig)

            # Exibição de tabela colorida com valores de forecast
            st.subheader("Tabela de Forecast (Valores Previstos)")
            forecast_only = forecast[forecast['type'] == 'Forecast'][["ds", "yhat", "yhat_lower", "yhat_upper"]]
            forecast_only.columns = ["Data", "Previsão (Yhat)", "Limite Inferior (Yhat Lower)", "Limite Superior (Yhat Upper)"]

            # Estilizando a tabela
            def color_forecast(val):
                if val.name == "Previsão (Yhat)":
                    return ['background-color: lightgreen' for _ in val]
                elif val.name == "Limite Inferior (Yhat Lower)":
                    return ['background-color: lightcoral' for _ in val]
                elif val.name == "Limite Superior (Yhat Upper)":
                    return ['background-color: lightblue' for _ in val]
                return ['' for _ in val]

            st.dataframe(
                forecast_only.style.apply(color_forecast, axis=0).format(
                    {"Previsão (Yhat)": "{:.2f}", "Limite Inferior (Yhat Lower)": "{:.2f}", "Limite Superior (Yhat Upper)": "{:.2f}"}
                )
            )

        else:
            st.warning("Selecione as colunas de data e valores para continuar.")

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
