import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import base64
from io import BytesIO

# Function to download DataFrame as CSV
def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast_results.csv">Download CSV File</a>'
    return href

# Function to create Plotly figure
def create_forecast_plot(forecast, col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast[forecast['type'] == 'Histórico']["ds"], y=forecast[forecast['type'] == 'Histórico']["y"], mode='lines', name="Histórico", line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=forecast[forecast['type'] == 'Forecast']["ds"], y=forecast[forecast['type'] == 'Forecast']["yhat"], mode='lines+markers', name="Previsão (yhat)", line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=forecast[forecast['type'] == 'Forecast']["ds"], y=forecast[forecast['type'] == 'Forecast']["yhat_upper"], mode='lines', name="Limite Superior", line=dict(color='orange', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=forecast[forecast['type'] == 'Forecast']["ds"], y=forecast[forecast['type'] == 'Forecast']["yhat_lower"], mode='lines', name="Limite Inferior", line=dict(color='red', width=1, dash='dash')))
    fig.update_layout(title=f"Forecast para {col}", xaxis_title="Data", yaxis_title="Valor", legend_title="Legenda", template="plotly_white")
    return fig

# Streamlit app
st.set_page_config(page_title="ARTEFACT - Análise Preditiva de P&L", layout="wide")

st.title("ARTEFACT")
st.subheader("Análise preditiva de P&L utilizando modelo de previsão Prophet")

# Sidebar for options
st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("Arraste e solte a base de dados aqui", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado. Use CSV, XLS ou XLSX.")
            st.stop()

        st.subheader("Dados Carregados (10 Primeiras Linhas):")
        st.dataframe(data.head(10))

        # Data preprocessing options
        st.sidebar.subheader("Pré-processamento")
        handle_missing = st.sidebar.checkbox("Tratar valores ausentes")
        if handle_missing:
            data = data.fillna(method='ffill').fillna(method='bfill')

        normalize_data = st.sidebar.checkbox("Normalizar dados")
        if normalize_data:
            scaler = MinMaxScaler()
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # Forecast configuration
        st.sidebar.subheader("Configuração do Forecast")
        date_column = st.sidebar.selectbox("Selecione a coluna de data:", data.columns)
        forecast_months = st.sidebar.slider("Meses para previsão:", 1, 24, 6)

        # Prophet parameters
        st.sidebar.subheader("Parâmetros do Prophet")
        yearly_seasonality = st.sidebar.checkbox("Sazonalidade anual", value=True)
        weekly_seasonality = st.sidebar.checkbox("Sazonalidade semanal", value=True)
        daily_seasonality = st.sidebar.checkbox("Sazonalidade diária", value=False)

        if date_column:
            data[date_column] = pd.to_datetime(data[date_column])
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            consolidated_forecast = pd.DataFrame()
            metrics = []

            for col in numeric_columns:
                with st.expander(f"Forecast para {col}"):
                    forecast_data = data[[date_column, col]].rename(columns={date_column: "ds", col: "y"})
                    
                    model = Prophet(yearly_seasonality=yearly_seasonality,
                                    weekly_seasonality=weekly_seasonality,
                                    daily_seasonality=daily_seasonality)
                    model.fit(forecast_data)

                    future = model.make_future_dataframe(periods=forecast_months, freq="M")
                    forecast = model.predict(future)

                    last_date = forecast_data['ds'].max()
                    forecast['y'] = forecast['ds'].map(dict(zip(forecast_data['ds'], forecast_data['y'])))
                    forecast['type'] = forecast['ds'].apply(lambda x: 'Histórico' if x <= last_date else 'Forecast')
                    forecast['y'] = forecast.apply(lambda row: row['y'] if row['type'] == 'Histórico' else row['yhat'], axis=1)
                    forecast['item'] = col

                    consolidated_forecast = pd.concat([consolidated_forecast, forecast], ignore_index=True)

                    # Calculate metrics
                    historical_values = forecast_data.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
                    mape = mean_absolute_percentage_error(historical_values['y'], historical_values['yhat'])
                    rmse = np.sqrt(mean_squared_error(historical_values['y'], historical_values['yhat']))
                    mae = mean_absolute_error(historical_values['y'], historical_values['yhat'])

                    metrics.append({
                        'Column': col,
                        'MAPE': mape,
                        'RMSE': rmse,
                        'MAE': mae
                    })

                    st.plotly_chart(create_forecast_plot(forecast, col), use_container_width=True)

                    st.write(f"MAPE: {mape:.2%}")
                    st.write(f"RMSE: {rmse:.2f}")
                    st.write(f"MAE: {mae:.2f}")

            # Display consolidated results
            st.subheader("Resultados Consolidados")
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df)

            st.subheader("Tabela Consolidada de Forecast")
            st.dataframe(consolidated_forecast[["ds", "item", "y", "yhat", "yhat_lower", "yhat_upper", "type"]])

            # Download options
            st.subheader("Download dos Resultados")
            st.markdown(download_csv(consolidated_forecast), unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
else:
    st.info("Por favor, faça o upload de um arquivo para começar.")