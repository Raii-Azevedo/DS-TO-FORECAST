import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error

# Título do App
st.title("ARTEFACT")

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
        st.subheader("Dados Carregados (10 Primeiras Linhas):")
        st.dataframe(data.head(10))

        # Tabela com diretrizes do MAPE
        st.subheader("Diretrizes para MAPE (Mean Absolute Percentage Error):")
        mape_guide = pd.DataFrame({
            "Intervalo do MAPE": ["< 10%", "10% - 20%", "20% - 50%", "> 50%"],
            "Classificação": ["Excelente previsão", "Boa previsão", "Previsão aceitável", "Previsão ruim"],
        })
        st.table(mape_guide)

        # Identificar colunas numéricas
        st.subheader("Configuração do Forecast")
        date_column = st.selectbox("Selecione a coluna de data:", data.columns)
        numeric_columns = data.select_dtypes(include=["number"]).columns

        if date_column and len(numeric_columns) > 0:
            # Converter coluna de data
            data[date_column] = pd.to_datetime(data[date_column], format="%d/%m/%Y")

            # Selecionar o número de meses para previsão
            forecast_months = st.slider(
                "Selecione quantos meses à frente deseja prever:",
                min_value=1,
                max_value=12,
                value=6,
                step=1
            )

            # Tabela consolidada para todos os forecasts
            consolidated_forecast = pd.DataFrame()

            for col in numeric_columns:
                st.subheader(f"Forecast para a Coluna: {col}")

                # Preparar os dados para o Prophet
                forecast_data = data[[date_column, col]].rename(
                    columns={date_column: "ds", col: "y"}
                )

                # Criar e treinar o modelo Prophet
                model = Prophet()
                model.fit(forecast_data)

                # Última data com valor
                last_date = forecast_data['ds'].max()

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

                # Adicionar coluna de origem (nome da coluna)
                forecast['item'] = col

                # Concatenar na tabela consolidada
                consolidated_forecast = pd.concat([consolidated_forecast, forecast], ignore_index=True)

                # Cálculo do MAPE
                historical_values = forecast_data.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
                mape = mean_absolute_percentage_error(historical_values['y'], historical_values['yhat'])
                st.write(f"MAPE para {col}: {mape * 100:.2f}%")

                # Gráfico de Forecast
                fig = go.Figure()

                # Linha de dados históricos
                fig.add_trace(go.Scatter(
                    x=forecast[forecast['type'] == 'Histórico']["ds"],
                    y=forecast[forecast['type'] == 'Histórico']["y"],
                    mode='lines',
                    name="Histórico",
                    line=dict(color='blue', width=2)
                ))

                # Linha de previsão
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
                    title=f"Forecast de {forecast_months} Meses para {col}",
                    xaxis_title="Data",
                    yaxis_title="Valor",
                    legend_title="Legenda",
                    template="plotly_white"
                )

                st.plotly_chart(fig)

            # Exibir tabela consolidada
            st.subheader("Tabela Consolidada de Forecast")
            st.dataframe(consolidated_forecast[["ds", "item", "y", "yhat", "yhat_lower", "yhat_upper", "type"]])

        else:
            st.warning("Selecione uma coluna de data e certifique-se de que existem colunas numéricas para previsão.")

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
