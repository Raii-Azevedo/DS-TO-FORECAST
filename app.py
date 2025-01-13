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
    # Verifica o tipo de arquivo e lê os dados
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):  # Suporte a arquivos Excel
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado. Use CSV, XLS ou XLSX.")
            st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
        st.stop()  # Para a execução aqui caso haja erro no arquivo

    # Exibição dos dados carregados
    st.subheader("Dados Carregados (Apenas 10 Primeiras Linhas):")
    st.dataframe(data.head(10))

    # Seleção de colunas
    st.subheader("Configuração do Forecast")
    date_column = st.selectbox("Selecione a coluna de data:", data.columns)
    value_column = st.selectbox("Selecione a coluna de valores:", data.columns)

    # Filtro para escolher o número de linhas históricas
    num_history = st.slider(
        "Selecione o número de registros históricos a exibir:",
        min_value=10,
        max_value=len(data),
        value=30,
        step=1
    )

    # Processa os dados apenas se ambas as colunas forem selecionadas
    if date_column and value_column:
        # Renomeia as colunas para o formato esperado pelo Prophet
        forecast_data = data[[date_column, value_column]].rename(
            columns={date_column: "ds", value_column: "y"}
        )

        # Verifica se a coluna 'y' está presente
        if 'y' not in forecast_data.columns:
            st.error("A coluna 'y' não foi encontrada nos dados históricos. Verifique o nome da coluna de valores.")
            st.stop()

        forecast_data["ds"] = pd.to_datetime(forecast_data["ds"])

        # Seleciona apenas o número de registros históricos solicitado
        forecast_data = forecast_data.tail(num_history)

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
        st.dataframe(forecast[["ds", "yhat_lower", "yhat", "yhat_upper", "type"]])

        # Criação do gráfico
        fig = go.Figure()

        # Linha de dados históricos
        fig.add_trace(go.Scatter(
            x=forecast_data["ds"],
            y=forecast_data["y"],
            mode='lines+markers',
            name="Histórico",
            line=dict(color='blue', width=1)
        ))

        # Linha superior (yhat_upper)
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode='lines',
            name="Limite Superior (yhat_upper)",
            line=dict(color='orange', width=1, dash='dash')
        ))

        # Linha de previsão (yhat)
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode='lines',
            name="Previsão (yhat)",
            line=dict(color='green', width=2)
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
        forecast_table.columns = ["Data", "Limite Inferior (Yhat Lower)", "Previsão (Yhat)", "Limite Superior (Yhat Upper)"]

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

        # Cálculo das métricas de erro (MAE, MSE, RMSE, MAPE)
        try:
            historical_forecast = forecast[forecast["type"] == "Histórico"]

            # Verificar se o número de registros históricos é suficiente
            st.write(f"Número de registros históricos: {len(historical_forecast)}")

            if len(historical_forecast) >= len(forecast_data):
                historical_forecast = historical_forecast.tail(len(forecast_data))  # Ajustar para o histórico correto
                st.write(f"Número de registros após ajuste: {len(historical_forecast)}")

                # Cálculo das métricas de erro
                mae = mean_absolute_error(historical_forecast["y"], historical_forecast["yhat"])
                mse = mean_squared_error(historical_forecast["y"], historical_forecast["yhat"])
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(historical_forecast["y"], historical_forecast["yhat"])

                st.subheader("Métricas de Acuracidade:")
                st.write(f"Erro Absoluto Médio (MAE): {mae:.2f}")
                st.write(f"Erro Quadrático Médio (MSE): {mse:.2f}")
                st.write(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
                st.write(f"Erro Percentual Absoluto Médio (MAPE): {mape * 100:.2f}%")
            else:
                st.warning("Não há dados históricos suficientes para calcular as métricas de erro.")
        except KeyError as e:
            st.error(f"Erro com as colunas: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro durante o cálculo das métricas: {e}")
