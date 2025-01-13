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

            # Processa os dados apenas se ambas as colunas forem selecionadas
            if date_column and value_column:
                # Renomeia as colunas para o formato esperado pelo Prophet
                forecast_data = data[[date_column, value_column]].rename(
                    columns={date_column: "ds", value_column: "y"}
                )
                forecast_data["ds"] = pd.to_datetime(forecast_data["ds"], format='%d/%m/%Y')

                # Usamos todo o histórico, então não há necessidade de filtrar
                # forecast_data = forecast_data.tail(num_history)

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

                # Cálculo do MAPE (Mean Absolute Percentage Error)
                if len(forecast_data) > 1:
                    # Considerando os dados mais recentes (último mês registrado) para o cálculo
                    historical_values = forecast_data.tail(forecast_months)
                    historical_values = historical_values.merge(forecast[['ds', 'yhat']], on='ds', how='left')
                    mape = mean_absolute_percentage_error(historical_values['y'], historical_values['yhat'])
                    st.subheader(f"MAPE (Mean Absolute Percentage Error): {mape * 100:.2f}%")

                # Exibição do resultado com a coluna extra
                st.subheader(f"Tabela do Forecast com Histórico e Forecast")
                st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "type"]])

                # Criação do gráfico
                fig = go.Figure()

                # Linha de dados históricos (agora uma linha contínua com marcadores)
                fig.add_trace(go.Scatter(
                    x=forecast_data["ds"],
                    y=forecast_data["y"],
                    mode='lines+markers',  # Linha contínua com marcadores
                    name="Histórico",
                    line=dict(color='blue', width=2),  # Linha azul contínua
                    marker=dict(size=6, color='blue')  # Marcadores azuis
                ))

                # Linha de previsão (yhat) com marcadores
                fig.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    mode='lines+markers',  # Linha contínua com marcadores
                    name="Previsão (yhat)",
                    line=dict(color='green', width=2),
                    marker=dict(size=6, color='green')  # Marcadores verdes
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

                # Filtrar apenas os dados de Forecast
                forecast_only = forecast[forecast['ds'] > forecast_data['ds'].max()]

                # Exibir uma tabela estilizada com os valores de Forecast
                st.subheader("Tabela de Forecast (Valores Previstos)")
                forecast_table = forecast_only[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                forecast_table.columns = ["Data", "Previsão (Yhat)", "Limite Inferior (Yhat Lower)", "Limite Superior (Yhat Upper)"]

                # Estilizando a tabela com cores
                def color_forecast(val):
                    if val.name == "Previsão (Yhat)":
                        return ['background-color: lightgreen' for _ in val]
                    elif val.name == "Limite Inferior (Yhat Lower)":
                        return ['background-color: lightcoral' for _ in val]
                    elif val.name == "Limite Superior (Yhat Upper)":
                        return ['background-color: lightblue' for _ in val]
                    return ['' for _ in val]

                st.dataframe(
                    forecast_table.style.apply(color_forecast, axis=0).format(
                        {"Previsão (Yhat)": "{:.2f}", "Limite Inferior (Yhat Lower)": "{:.2f}", "Limite Superior (Yhat Upper)": "{:.2f}"}
                    )
                )

                # Exibir um gráfico adicional para os Forecasts
                st.subheader("Gráfico de Valores de Forecast")
                forecast_fig = go.Figure()

                # Adicionar as barras para Yhat, Yhat Lower e Yhat Upper
                forecast_fig.add_trace(go.Bar(
                    x=forecast_only["ds"],
                    y=forecast_only["yhat"],
                    name="Previsão (Yhat)",
                    marker_color='lightgreen'
                ))

                forecast_fig.add_trace(go.Bar(
                    x=forecast_only["ds"],
                    y=forecast_only["yhat_lower"],
                    name="Limite Inferior (Yhat Lower)",
                    marker_color='lightcoral'
                ))

                forecast_fig.add_trace(go.Bar(
                    x=forecast_only["ds"],
                    y=forecast_only["yhat_upper"],
                    name="Limite Superior (Yhat Upper)",
                    marker_color='lightblue'
                ))

                # Configuração do layout do gráfico
                forecast_fig.update_layout(
                    title="Valores de Forecast",
                    xaxis_title="Data",
                    yaxis_title="Valor",
                    barmode="group",
                    template="plotly_white"
                )

                st.plotly_chart(forecast_fig)

            else:
                st.warning("Selecione as colunas de data e valores para continuar.")

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
