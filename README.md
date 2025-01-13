
# P&L - ARTEFACT

Este projeto é um aplicativo de previsão de lucros e perdas (P&L) usando Streamlit e Prophet. Ele permite carregar dados históricos, aplicar o modelo Prophet para previsão, visualizar os resultados em gráficos interativos e exibir tabelas de valores históricos e previstos.

---

## Objetivo

O objetivo deste projeto é oferecer uma ferramenta acessível para prever tendências financeiras futuras com base em dados históricos, permitindo que usuários tomem decisões informadas.

---

## Estrutura do Código

O código está organizado para carregar dados, processá-los, aplicar o modelo de previsão Prophet e exibir os resultados.

---

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- Bibliotecas necessárias:
  - `streamlit`
  - `pandas`
  - `prophet`
  - `plotly`
  - `scikit-learn`
  - `numpy`
  
Para instalar as dependências, use:
```bash
pip install streamlit pandas prophet plotly scikit-learn numpy
```

---

## Funcionalidades

1. **Upload de Arquivo**
   - Aceita arquivos nos formatos CSV, XLS e XLSX.
   - Mostra as primeiras 10 linhas dos dados carregados.

2. **Seleção de Colunas**
   - Permite selecionar a coluna de data e a coluna de valores.

3. **Forecast com Prophet**
   - Gera previsões com base nos dados históricos.
   - Adiciona limites superior e inferior para o intervalo de confiança.

4. **Cálculo de MAPE**
   - Calcula o Mean Absolute Percentage Error (MAPE) para medir a precisão do modelo.

5. **Visualização**
   - Gráfico interativo para histórico e previsão com intervalos de confiança.
   - Tabelas estilizadas para os dados previstos.

---

## Passo a Passo

### 1. Carregar Dados

O usuário carrega um arquivo arrastando-o para o app. O código detecta automaticamente o formato do arquivo e o processa.

### 2. Seleção de Colunas

Após o carregamento, o usuário escolhe as colunas correspondentes à data e aos valores.

### 3. Previsão

O modelo Prophet é aplicado:
- Ele usa todos os dados históricos fornecidos.
- Calcula previsões para o número de meses desejados.

### 4. Exibição dos Resultados

- Gráficos interativos mostram o histórico, previsões e intervalos de confiança.
- Tabelas coloridas destacam os valores previstos.

---

## Código

```python
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
    try:
        # Carregar o arquivo
        data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.subheader("Dados Carregados (Apenas 10 Primeiras Linhas):")
        st.dataframe(data.head(10))

        # Seleção de colunas
        date_column = st.selectbox("Selecione a coluna de data:", data.columns)
        value_column = st.selectbox("Selecione a coluna de valores:", data.columns)

        if date_column and value_column:
            forecast_data = data[[date_column, value_column]].rename(columns={date_column: "ds", value_column: "y"})
            forecast_data["ds"] = pd.to_datetime(forecast_data["ds"], format="%d/%m/%Y")
            
            model = Prophet()
            model.fit(forecast_data)
            
            last_date = forecast_data['ds'].max()
            forecast_months = st.slider("Selecione quantos meses à frente deseja prever:", min_value=1, max_value=12, value=6, step=1)
            
            future = model.make_future_dataframe(periods=forecast_months, freq="M")
            forecast = model.predict(future)
            
            forecast = forecast.merge(forecast_data, on="ds", how="left")
            forecast['y'] = forecast['y'].fillna(forecast['yhat'])
            
            st.subheader("MAPE (Mean Absolute Percentage Error)")
            mape = mean_absolute_percentage_error(forecast_data['y'], forecast['yhat'].dropna())
            st.write(f"{mape * 100:.2f}%")

            # Gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_data["ds"], y=forecast_data["y"], mode='lines', name="Histórico"))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name="Previsão"))
            st.plotly_chart(fig)

            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    except Exception as e:
        st.error(f"Erro: {e}")
```

---

## Uso

1. Inicie o app com o comando:
   ```bash
   streamlit run app.py
   ```

2. Carregue o arquivo de dados.

3. Selecione as colunas e configure os parâmetros.

4. Visualize os resultados.

---

## Contribuições

Sinta-se à vontade para sugerir melhorias ou abrir issues. 😊
